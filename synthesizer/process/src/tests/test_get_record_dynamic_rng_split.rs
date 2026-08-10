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

//! Reproduction for a second non-deterministic deployment-verification vector, via `get.record.dynamic`.
//!
//! `get.record.dynamic`'s helper `compute_or_sample_path`
//! (`synthesizer/program/src/logic/instruction/operation/get_record_dynamic.rs`) samples an
//! arbitrary dummy entry value from the *ambient* `rand::rng()` on the not-present branch. That
//! branch is reached during `CheckDeployment` synthesis: a sampled dynamic-record input carries no
//! data (`Stack::sample_dynamic_record` builds it with `data = None`), so every `get.record.dynamic`
//! on it hits the sampling branch. The sampled value is stored to a register, and a program can feed
//! it into a `call.dynamic` target — selecting a closure hard-errors, selecting a missing function
//! is tolerated in dummy mode — so the same deployment verifies differently per validator.
//!
//! This is an **independent** source of the fork described in `test_deploy_rng_split.rs`: this
//! program never reads `self.signer`, so it splits even with the burner drawn from a seeded RNG.
//! The test verifies one deployment under 40 different ambient RNGs and asserts they all agree; it
//! fails on the vulnerable code and passes once the sampling is made deterministic.

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
fn test_get_record_dynamic_ambient_rng_forks_deployment_verification() {
    let program_name = "dynsplit";
    let network_field = ident_field("aleo");
    let program_field = ident_field(program_name);
    // `pick` is a *closure* => selecting it hard-errors in dummy mode.
    let closure_field = ident_field("pick");
    // `zzz` is not defined => selecting it is tolerated (Ok(None)) in dummy mode.
    let missing_field = ident_field("zzz");

    // `main` reads an entry from a dynamic-record input (a random value in dummy/None mode), derives
    // one bit from it, and uses that bit to pick the dynamic-call target:
    //   bit == 0  -> `pick` (closure)  -> verification bails
    //   bit == 1  -> `zzz`  (missing)  -> verification succeeds
    let src = format!(
        r"
program {program_name}.aleo;

closure pick:
    input r0 as u8;
    add r0 r0 into r1;
    output r1 as u8;

function main:
    input r0 as dynamic.record;
    get.record.dynamic r0.secret into r1 as u8;
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

    // Deploy the program. As with the burner vector, `deploy()` synthesizes under the same
    // non-determinism, so a successful deploy corresponds to the "missing function" branch. Retry
    // across seeds until one succeeds, then hold that single deployment fixed.
    let deployment = (0u64..256)
        .find_map(|seed| stack.deploy::<CurrentAleo, _>(&mut ChaChaRng::seed_from_u64(seed)).ok())
        .expect("failed to synthesize a deployment in 256 attempts");

    // Verify the SAME byte-identical deployment under many different ambient RNGs, modelling distinct
    // validators each calling the verify path with its own `rand::rng()`.
    let consensus_version = ConsensusVersion::V16;
    let mut accepted = 0usize;
    let mut rejected = 0usize;
    for seed in 0u64..40 {
        let mut ambient_rng = ChaChaRng::seed_from_u64(0xD1CE ^ seed);
        match stack.verify_deployment::<CurrentAleo, _>(consensus_version, &deployment, &mut ambient_rng) {
            Ok(()) => accepted += 1,
            Err(_) => rejected += 1,
        }
    }

    println!(
        "verify_deployment on the SAME deployment across 40 ambient RNGs: {accepted} accepted, {rejected} rejected"
    );

    // Deployment verification MUST be deterministic. `get.record.dynamic`'s dummy sampling must not
    // let the ambient RNG change the verification outcome.
    //
    // Before the fix (dummy value from `rand::rng()`), this splits ~50/50 and forks the chain.
    // After the fix (dummy value from a deterministic RNG), all outcomes agree.
    assert!(
        accepted == 0 || rejected == 0,
        "non-deterministic deployment verification via get.record.dynamic: {accepted} accepted vs {rejected} rejected"
    );
}
