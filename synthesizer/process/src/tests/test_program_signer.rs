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
