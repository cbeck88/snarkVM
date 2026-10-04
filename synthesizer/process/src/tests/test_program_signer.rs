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

//! Tests for program-owned signers (exploratory prototype).

use crate::CallStack;
use circuit::network::AleoV0;
use console::{
    account::PrivateKey,
    network::{MainnetV0, prelude::*},
    program::{
        Identifier,
        Literal,
        PROGRAM_SIGNER_KIND_SIGNER,
        Plaintext,
        ProgramSigner,
        Value,
        program_signer_tweak,
        program_signer_tweak_domain,
    },
    types::Scalar,
};
use snarkvm_synthesizer_program::Program;

type CurrentNetwork = MainnetV0;
type CurrentAleo = AleoV0;

/// Returns Aleo instructions that compute `X = x * G`, `t = Ht(signer, self, X)`, and `Y = X + t * G`
/// from the internal secret `x` in register `r{x}`, using the registers `r{start}..r{start + 7}`.
/// `X` is in `r{start}`, `t` in `r{start + 5}`, and `Y` in `r{start + 7}`.
pub(crate) fn tweak_instructions<N: Network>(program_name: &str, x: usize, start: usize) -> String {
    let domain = program_signer_tweak_domain::<N>();
    let r = |i: usize| format!("r{}", start + i);
    format!(
        r"
    mul aleo::GENERATOR r{x} into {r0};
    cast {r0} into {r1} as group.x;
    cast {r0} into {r2} as group.y;
    cast {program_name}.aleo into {r3} as field;
    cast {domain} {PROGRAM_SIGNER_KIND_SIGNER}field {r3} {r1} {r2} into {r4} as [field; 5u32];
    hash.psd4 {r4} into {r5} as scalar;
    mul aleo::GENERATOR {r5} into {r6};
    add {r0} {r6} into {r7};",
        r0 = r(0),
        r1 = r(1),
        r2 = r(2),
        r3 = r(3),
        r4 = r(4),
        r5 = r(5),
        r6 = r(6),
        r7 = r(7),
    )
}

#[test]
fn test_program_signer_tweak_matches_aleo_instructions() {
    let rng = &mut TestRng::default();

    let program = Program::<CurrentNetwork>::from_str(&format!(
        r"
program vault_tweak.aleo;

function tweak:
    input r0 as scalar.private;
    {}
    output r6 as scalar.private;
    output r8 as group.private;
",
        tweak_instructions::<CurrentNetwork>("vault_tweak", 0, 1)
    ))
    .unwrap();
    let process = crate::test_helpers::sample_process(&program);
    let function_name = Identifier::from_str("tweak").unwrap();

    for _ in 0..4 {
        let x = Scalar::<CurrentNetwork>::rand(rng);
        let inputs = [Value::<CurrentNetwork>::from(Literal::Scalar(x))];
        let private_key = PrivateKey::<CurrentNetwork>::new(rng).unwrap();
        let authorization =
            process.authorize::<CurrentAleo, _>(&private_key, program.id(), function_name, inputs.iter(), rng).unwrap();
        let stack = process.get_stack(program.id()).unwrap();
        let response = stack
            .evaluate_function::<CurrentAleo, _>(CallStack::evaluate(authorization).unwrap(), None, None, rng)
            .unwrap();

        // The native tweak and address.
        let internal = CurrentNetwork::g_scalar_multiply(&x);
        let tweak =
            program_signer_tweak(PROGRAM_SIGNER_KIND_SIGNER, &program.id().to_address().unwrap(), &internal).unwrap();
        let signer = ProgramSigner::from_internal_secret(*program.id(), x).unwrap();

        assert_eq!(response.outputs()[0], Value::Plaintext(Plaintext::from(Literal::Scalar(tweak))));
        assert_eq!(response.outputs()[1], Value::Plaintext(Plaintext::from(Literal::Group(**signer.address()))));
    }
}

/// Synthesizes `program_id/function_name` in `CheckDeployment` mode with the given request signer,
/// as the root (`caller == None`) or as a child of `caller`, and returns
/// `(public, private, constraints, nonzeros)` of the function circuit.
fn measure_function(
    process: &crate::Process<CurrentNetwork>,
    program_id: &str,
    function_name: &str,
    signer: &console::program::RequestSigner<CurrentNetwork>,
    caller: Option<&str>,
    rng: &mut TestRng,
) -> (u64, u64, u64, (u64, u64, u64)) {
    use console::program::{ProgramID, Request};
    use snarkvm_synthesizer_program::StackTrait;

    let program_id = ProgramID::<CurrentNetwork>::from_str(program_id).unwrap();
    let function_name = Identifier::<CurrentNetwork>::from_str(function_name).unwrap();
    let stack = process.get_stack(program_id).unwrap();
    let input_types = stack.get_function(&function_name).unwrap().input_types();
    let address = signer.address().unwrap();
    // In `CheckDeployment` mode, child requests are signed by the burner key, so external records belong to it.
    let burner = PrivateKey::<CurrentNetwork>::new(rng).unwrap();
    let burner_address = console::account::Address::try_from(&burner).unwrap();
    let inputs = input_types
        .iter()
        .map(|input_type| match input_type {
            console::program::ValueType::ExternalRecord(locator) => process
                .get_stack(locator.program_id())
                .unwrap()
                .sample_value(&burner_address, &console::program::ValueType::Record(*locator.resource()).into(), rng),
            _ => stack.sample_value(&address, &input_type.into(), rng),
        })
        .collect::<Result<Vec<_>>>()
        .unwrap();
    let caller = caller.map(|caller| ProgramID::<CurrentNetwork>::from_str(caller).unwrap());
    let root_tvk = caller.map(|_| console::types::Field::rand(rng));
    let request: Request<CurrentNetwork> = signer
        .sign(program_id, function_name, inputs.into_iter(), &input_types, root_tvk, caller.is_none(), None, false, rng)
        .unwrap();
    let assignments = std::sync::Arc::new(parking_lot::RwLock::new(Vec::new()));
    let call_stack = CallStack::CheckDeployment(vec![request], burner, assignments.clone(), None, None, None);
    stack.execute_function::<CurrentAleo, _>(call_stack, caller, root_tvk, rng).unwrap();
    let assignments = assignments.read();
    // The root's assignment is pushed last, after its children's.
    let (assignment, _) = assignments.last().unwrap();
    (assignment.num_public(), assignment.num_private(), assignment.num_constraints(), assignment.num_nonzeros())
}

/// Prints the circuit sizes of `credits.aleo` functions with ordinary and program signers.
/// Run with and without the `program-signer` feature to compare.
#[test]
fn test_program_signer_measure_credits() {
    let rng = &mut TestRng::default();
    let process = crate::Process::<CurrentNetwork>::load().unwrap();
    let account = console::program::RequestSigner::Account(PrivateKey::new(rng).unwrap());
    let vault = console::program::ProgramID::<CurrentNetwork>::from_str("vault.aleo").unwrap();
    let program_signer = console::program::RequestSigner::Program(
        ProgramSigner::from_internal_secret(vault, Scalar::rand(rng)).unwrap(),
    );
    let feature = if cfg!(feature = "program-signer") { "on" } else { "off" };

    for function_name in ["transfer_private", "transfer_public", "join", "transfer_private_to_public"] {
        let root = measure_function(&process, "credits.aleo", function_name, &account, None, rng);
        let child = measure_function(&process, "credits.aleo", function_name, &account, Some("vault.aleo"), rng);
        println!(
            "MEASURE feature={feature} credits.aleo/{function_name} ordinary root  (public, private, constraints, nonzeros) = {root:?}"
        );
        println!(
            "MEASURE feature={feature} credits.aleo/{function_name} ordinary child (public, private, constraints, nonzeros) = {child:?}"
        );
        if cfg!(feature = "program-signer") {
            let program =
                measure_function(&process, "credits.aleo", function_name, &program_signer, Some("vault.aleo"), rng);
            println!(
                "MEASURE feature={feature} credits.aleo/{function_name} program  child (public, private, constraints, nonzeros) = {program:?}"
            );
            assert_eq!(child, program);
        }
    }
}

/// Prints the circuit size of an opted-in vault function (which pays for the root tweak check), with ordinary and
/// program signers at the root. Run with and without the `program-signer` feature to compare.
#[test]
fn test_program_signer_measure_vault() {
    let rng = &mut TestRng::default();
    let program = Program::<CurrentNetwork>::from_str(
        r"
import credits.aleo;

program vault_measure.aleo;

struct approval:
    vault as address;
    nonce as u64;
    recipient as address;
    amount as u64;

record config:
    owner as address.private;
    signer0 as address.private;
    signer1 as address.private;
    signer2 as address.private;
    threshold as u8.private;
    nonce as u64.private;

function execute_transfer:
    input r0 as config.record;
    input r1 as credits.aleo/credits.record;
    input r2 as address.private;
    input r3 as u64.private;
    input r4 as signature.private;
    input r5 as signature.private;
    input r6 as signature.private;
    assert.eq self.caller self.signer;
    cast self.signer r0.nonce r2 r3 into r7 as approval;
    sign.verify r4 r0.signer0 r7 into r8;
    sign.verify r5 r0.signer1 r7 into r9;
    sign.verify r6 r0.signer2 r7 into r10;
    ternary r8 1u8 0u8 into r11;
    ternary r9 1u8 0u8 into r12;
    ternary r10 1u8 0u8 into r13;
    add r11 r12 into r14;
    add r14 r13 into r15;
    gte r15 r0.threshold into r16;
    assert.eq r16 true;
    call credits.aleo/transfer_private r1 r2 r3 into r17 r18;
    add r0.nonce 1u64 into r19;
    cast r0.owner r0.signer0 r0.signer1 r0.signer2 r0.threshold r19 into r20 as config.record;
    output r20 as config.record;
    output r17 as credits.aleo/credits.record;
    output r18 as credits.aleo/credits.record;

function noop:
    input r0 as u64.private;
    output r0 as u64.private;
",
    )
    .unwrap();
    let process = crate::test_helpers::sample_process(&program);
    let account = console::program::RequestSigner::Account(PrivateKey::new(rng).unwrap());
    let program_signer = console::program::RequestSigner::Program(
        ProgramSigner::from_internal_secret(*program.id(), Scalar::rand(rng)).unwrap(),
    );
    let feature = if cfg!(feature = "program-signer") { "on" } else { "off" };
    for function_name in ["execute_transfer", "noop"] {
        let root = measure_function(&process, "vault_measure.aleo", function_name, &account, None, rng);
        println!(
            "MEASURE feature={feature} vault_measure.aleo/{function_name} ordinary root (public, private, constraints, nonzeros) = {root:?}"
        );
        if cfg!(feature = "program-signer") {
            let program = measure_function(&process, "vault_measure.aleo", function_name, &program_signer, None, rng);
            println!(
                "MEASURE feature={feature} vault_measure.aleo/{function_name} program  root (public, private, constraints, nonzeros) = {program:?}"
            );
            assert_eq!(root, program);
        }
    }
}
