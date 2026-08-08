// Copyright (c) 2019-2026 Provable Inc.
// This file is part of the snarkVM library.

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at:

// http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! Reproduction for the non-deterministic deployment-verification bug.
//!
//! `Stack::verify_deployment` seeds its input/synthesis RNGs deterministically from the
//! deployment ID, but samples each function's dummy "burner" private key from the *ambient*
//! `rng` argument (see `deploy.rs`: `let burner_private_key = PrivateKey::new(rng)?;`). The
//! burner sets `self.signer` for the dummy `CheckDeployment` synthesis. `call.dynamic` lets a
//! program branch its dynamic-call target on `self.signer`: selecting a **closure** hard-errors
//! (`Cannot dynamically evaluate a closure`), while selecting a **missing function** is tolerated
//! in dummy mode. So the *same* byte-identical deployment verifies differently depending only on
//! the ambient RNG the caller passes in — which snarkOS derives from `rand::rng()` per validator.
//!
//! This test deploys one such program, then verifies the *same* deployment under many different
//! ambient RNGs and asserts they all agree. It fails on the vulnerable code (the outcomes split
//! ~50/50) and passes once the burner is drawn from the deployment-ID-seeded RNG.

use circuit::network::AleoV0;
use console::{
    network::{MainnetV0, prelude::*},
    program::{Identifier, ProgramID},
    types::Field,
};
use snarkvm_synthesizer_program::Program;

use crate::Process;

use rand::SeedableRng;
use rand_chacha::ChaChaRng;

type CurrentNetwork = MainnetV0;
type CurrentAleo = AleoV0;

/// Field encoding of an identifier, as it must appear literally inside a `call.dynamic` operand.
fn ident_field(name: &str) -> Field<CurrentNetwork> {
    Identifier::<CurrentNetwork>::from_str(name).unwrap().to_field().unwrap()
}

#[test]
fn test_ambient_rng_burner_forks_deployment_verification() {
    // The program name, and the identifiers used to build the `call.dynamic` operands.
    let program_name = "splitter";
    let network_field = ident_field("aleo");
    let program_field = ident_field(program_name);
    // `pick` is a *closure* in this program => selecting it hard-errors in dummy mode.
    let closure_field = ident_field("pick");
    // `zzz` is not defined => selecting it is tolerated (Ok(None)) in dummy mode.
    let missing_field = ident_field("zzz");

    // `main` derives one bit from `self.signer` and uses it to pick the dynamic-call target:
    //   bit == 0  -> `pick`  (closure)  -> verification bails
    //   bit == 1  -> `zzz`   (missing)  -> verification succeeds
    let src = format!(
        r"
program {program_name}.aleo;

closure pick:
    input r0 as u8;
    add r0 r0 into r1;
    output r1 as u8;

function main:
    cast.lossy self.signer into r0 as field;
    cast.lossy r0 into r1 as u8;
    and r1 1u8 into r2;
    is.eq r2 0u8 into r3;
    ternary r3 {closure_field} {missing_field} into r4;
    call.dynamic {program_field} {network_field} r4 with r1 (as u8.private) into r5 (as u8.private);
    output r5 as u8.private;"
    );

    let (rest, program) = Program::<CurrentNetwork>::parse(&src).unwrap();
    assert!(rest.is_empty(), "parser did not consume the whole program: {rest:?}");

    let process = Process::<CurrentNetwork>::load().unwrap();
    process.lock().add_program(&program).unwrap();
    let stack = process.get_stack(ProgramID::from_str(&format!("{program_name}.aleo")).unwrap()).unwrap();

    // Deploy the program. `deploy()` synthesizes keys under the *same* ambient-RNG non-determinism,
    // so a successful deploy corresponds to the "missing function" synthesis branch. Retry across
    // seeds until one succeeds, then hold that single deployment fixed for the rest of the test.
    let deployment = (0u64..256)
        .find_map(|seed| stack.deploy::<CurrentAleo, _>(&mut ChaChaRng::seed_from_u64(seed)).ok())
        .expect("failed to synthesize a deployment in 256 attempts");

    // Now verify that ONE fixed, byte-identical deployment yields divergent verification results
    // depending only on the ambient RNG the caller passes in. This models distinct validators, each
    // calling the verify path with its own `rand::rng()`.
    let consensus_version = ConsensusVersion::V16;
    let mut accepted = 0usize;
    let mut rejected = 0usize;
    for seed in 0u64..40 {
        // A fresh ambient RNG per "validator". Only the burner key is drawn from this.
        let mut ambient_rng = ChaChaRng::seed_from_u64(0xA11CE ^ seed);
        match stack.verify_deployment::<CurrentAleo, _>(consensus_version, &deployment, &mut ambient_rng) {
            Ok(()) => accepted += 1,
            Err(_) => rejected += 1,
        }
    }

    println!(
        "verify_deployment on the SAME deployment across 40 ambient RNGs: {accepted} accepted, {rejected} rejected"
    );

    // Deployment verification MUST be deterministic: every validator, regardless of the ambient RNG
    // it passes in, must reach the same accept/reject decision on a byte-identical deployment.
    //
    // Before the fix (burner drawn from the ambient `rng`), this splits ~50/50 and forks the chain.
    // After the fix (burner drawn from the deployment-ID-seeded `seeded_rng`), all outcomes agree.
    assert!(
        accepted == 0 || rejected == 0,
        "non-deterministic deployment verification: {accepted} accepted vs {rejected} rejected across ambient RNGs"
    );
}
