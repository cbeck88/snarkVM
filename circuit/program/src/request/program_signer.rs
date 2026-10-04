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

use super::*;

/// The program signer witness of a request circuit (exploratory prototype).
///
/// Every request circuit carries this witness when the `program-signer` feature is enabled, so that
/// the circuit has the same shape for ordinary and program signers. For an ordinary signer,
/// `is_program` is `false` and the other values are unused placeholders.
pub struct ProgramSignerWitness<A: Aleo> {
    /// A flag indicating whether the signer is a program-owned address.
    pub(super) is_program: Boolean<A>,
    /// The internal key `X`.
    pub(super) internal: Group<A>,
    /// The view key `y` of the program-owned address.
    pub(super) view_key: Scalar<A>,
    /// The transition secret key `tsk`.
    pub(super) tsk: Scalar<A>,
}

impl<A: Aleo> ProgramSignerWitness<A> {
    /// Injects the program signer witness of the given request in the given mode.
    pub(super) fn new(mode: Mode, witness: Option<&console::ProgramSignerWitness<A::Network>>) -> Self {
        match witness {
            Some(witness) => Self {
                is_program: Boolean::new(mode, true),
                internal: Group::new(mode, *witness.internal()),
                view_key: Scalar::new(mode, *witness.view_key()),
                tsk: Scalar::new(mode, *witness.tsk()),
            },
            None => Self {
                is_program: Boolean::new(mode, false),
                internal: Group::new(mode, console::Group::generator()),
                view_key: Scalar::new(mode, console::Scalar::zero()),
                tsk: Scalar::new(mode, console::Scalar::zero()),
            },
        }
    }

    /// Ejects the program signer witness, if the signer is a program-owned address.
    pub(super) fn eject_value(&self) -> Option<console::ProgramSignerWitness<A::Network>> {
        match self.is_program.eject_value() {
            true => Some(console::ProgramSignerWitness::new(
                console::PROGRAM_SIGNER_KIND_SIGNER,
                self.internal.eject_value(),
                self.view_key.eject_value(),
                self.tsk.eject_value(),
            )),
            false => None,
        }
    }

    /// Ejects the mode of the program signer witness.
    pub(super) fn eject_mode(&self) -> Mode {
        Mode::combine(self.is_program.eject_mode(), [
            self.internal.eject_mode(),
            self.view_key.eject_mode(),
            self.tsk.eject_mode(),
        ])
    }
}

/// Returns the program signer tweak `Ht(kind, P, X)` over the plaintext `[domain, kind, P.x, X.x, X.y]`,
/// matching `console::program_signer_tweak`.
pub(super) fn program_signer_tweak<A: Aleo>(kind: u8, program_address: &Address<A>, internal: &Group<A>) -> Scalar<A> {
    let elements = [
        Field::constant(console::program_signer_tweak_domain::<A::Network>()),
        Field::constant(console::Field::from_u8(kind)),
        program_address.to_field(),
        internal.to_x_coordinate(),
        internal.to_y_coordinate(),
    ];
    let preimage = Plaintext::Array(
        elements.into_iter().map(|f| Plaintext::Literal(Literal::Field(f), Default::default())).collect(),
        Default::default(),
    );
    A::hash_to_scalar_psd4(&preimage.to_fields())
}

#[cfg(all(test, feature = "program-signer"))]
mod tests {
    use super::*;
    use crate::Circuit;
    use snarkvm_utilities::{TestRng, Uniform};

    type CurrentNetwork = <Circuit as Environment>::Network;

    /// The counts `(constants, public, private, constraints)` of a request verification.
    type Counts = (u64, u64, u64, u64);

    /// Returns the inputs and input types of a test function, with or without a record input owned by `owner`.
    fn sample_inputs(
        owner: console::Address<CurrentNetwork>,
        use_record: bool,
    ) -> (Vec<console::Value<CurrentNetwork>>, Vec<console::ValueType<CurrentNetwork>>) {
        let record = format!(
            "{{ owner: {owner}.private, token_amount: 100u64.private, _nonce: 0group.public, _version: 1u8.public }}"
        );
        let mut inputs = vec![console::Value::from_str("5u64").unwrap(), console::Value::from_str("6u64").unwrap()];
        let mut input_types = vec![
            console::ValueType::from_str("u64.public").unwrap(),
            console::ValueType::from_str("u64.private").unwrap(),
        ];
        if use_record {
            inputs.push(console::Value::from_str(&record).unwrap());
            input_types.push(console::ValueType::from_str("token.record").unwrap());
        }
        (inputs, input_types)
    }

    /// Returns a request signed by an ordinary account.
    fn sample_ordinary(
        program_id: &str,
        use_record: bool,
        is_root: bool,
        rng: &mut TestRng,
    ) -> (console::Request<CurrentNetwork>, Vec<console::ValueType<CurrentNetwork>>) {
        let private_key = snarkvm_console_account::PrivateKey::new(rng).unwrap();
        let address = console::Address::try_from(&private_key).unwrap();
        let (inputs, input_types) = sample_inputs(address, use_record);
        let request = console::Request::sign(
            &private_key,
            console::ProgramID::from_str(program_id).unwrap(),
            console::Identifier::from_str("foo").unwrap(),
            inputs.iter(),
            &input_types,
            None,
            is_root,
            None,
            false,
            rng,
        )
        .unwrap();
        (request, input_types)
    }

    /// Returns a request whose signer is the given program signer.
    fn sample_program(
        signer: &console::ProgramSigner<CurrentNetwork>,
        program_id: &str,
        use_record: bool,
        is_root: bool,
        rng: &mut TestRng,
    ) -> (console::Request<CurrentNetwork>, Vec<console::ValueType<CurrentNetwork>>) {
        let (inputs, input_types) = sample_inputs(*signer.address(), use_record);
        let request = console::Request::sign_as_program(
            signer,
            console::ProgramID::from_str(program_id).unwrap(),
            console::Identifier::from_str("foo").unwrap(),
            inputs.iter(),
            &input_types,
            None,
            is_root,
            None,
            false,
            rng,
        )
        .unwrap();
        (request, input_types)
    }

    /// Returns a vault signer for `vault.aleo`.
    fn sample_vault_signer(rng: &mut TestRng) -> console::ProgramSigner<CurrentNetwork> {
        let vault = console::ProgramID::from_str("vault.aleo").unwrap();
        console::ProgramSigner::from_internal_secret(vault, console::Scalar::rand(rng)).unwrap()
    }

    /// Verifies the request in a fresh circuit, and returns the result and the counts in scope.
    fn verify_in_circuit(
        request: &console::Request<CurrentNetwork>,
        input_types: &[console::ValueType<CurrentNetwork>],
        tpk: Option<console::Group<CurrentNetwork>>,
        is_root: bool,
        parent: &str,
    ) -> (bool, Counts) {
        Circuit::reset();
        let parent_id = console::ProgramID::<CurrentNetwork>::from_str(parent).unwrap();
        let is_opted_in = console::program_signer_opt_in(&parent_id) && is_root;
        let tpk = Group::<Circuit>::new(Mode::Public, tpk.unwrap_or_else(|| request.to_tpk()));
        let parent = Address::<Circuit>::new(Mode::Public, parent_id.to_address().unwrap());
        let is_root = Boolean::<Circuit>::new(Mode::Public, is_root);
        let root_tvk = Field::<Circuit>::new(Mode::Private, *request.tvk());
        let result = Circuit::scope("verify", || {
            let request = Request::<Circuit>::new(Mode::Private, request.clone());
            let candidate = request.verify(input_types, &tpk, Some(root_tvk), is_root, None, &parent, is_opted_in);
            let counts = (
                Circuit::num_constants_in_scope(),
                Circuit::num_public_in_scope(),
                Circuit::num_private_in_scope(),
                Circuit::num_constraints_in_scope(),
            );
            (candidate.eject_value() && Circuit::is_satisfied_in_scope(), counts)
        });
        Circuit::reset();
        result
    }

    /// Rebuilds the request with the given components replaced.
    #[allow(clippy::type_complexity)]
    fn rebuild(
        request: &console::Request<CurrentNetwork>,
        input_ids: Vec<console::InputID<CurrentNetwork>>,
        sk_tag: console::Field<CurrentNetwork>,
        tvk: console::Field<CurrentNetwork>,
        tcm: console::Field<CurrentNetwork>,
        program_signer: Option<console::ProgramSignerWitness<CurrentNetwork>>,
    ) -> console::Request<CurrentNetwork> {
        console::Request::from((
            *request.signer(),
            *request.network_id(),
            *request.program_id(),
            *request.function_name(),
            input_ids,
            request.inputs().to_vec(),
            *request.signature(),
            sk_tag,
            tvk,
            tcm,
            *request.scm(),
            request.is_dynamic(),
        ))
        .with_program_signer(program_signer)
    }

    #[test]
    fn test_program_signer_circuit_shape_and_counts() {
        let rng = &mut TestRng::default();
        let signer = sample_vault_signer(rng);
        // Warm up the lazily-initialized constants, so that the constant counts below are comparable.
        for use_record in [false, true] {
            let (warm_up, input_types) = sample_program(&signer, "vault.aleo", use_record, true, rng);
            assert!(verify_in_circuit(&warm_up, &input_types, None, true, "vault.aleo").0);
        }
        for use_record in [false, true] {
            // At the root of an opted-in program.
            let (ordinary, input_types) = sample_ordinary("vault.aleo", use_record, true, rng);
            let (ordinary_ok, ordinary_counts) = verify_in_circuit(&ordinary, &input_types, None, true, "vault.aleo");
            let (program, input_types) = sample_program(&signer, "vault.aleo", use_record, true, rng);
            let (program_ok, program_counts) = verify_in_circuit(&program, &input_types, None, true, "vault.aleo");
            assert!(ordinary_ok && program_ok);
            assert_eq!(ordinary_counts, program_counts, "use_record = {use_record}, opted in");
            println!("opted-in root, use_record = {use_record}: {program_counts:?}");

            // As a child, in a program that has not opted in (e.g. 'credits.aleo').
            let (ordinary, input_types) = sample_ordinary("credits.aleo", use_record, false, rng);
            let (ordinary_ok, ordinary_counts) = verify_in_circuit(&ordinary, &input_types, None, false, "vault.aleo");
            let (program, input_types) = sample_program(&signer, "credits.aleo", use_record, false, rng);
            let (program_ok, program_counts) = verify_in_circuit(&program, &input_types, None, false, "vault.aleo");
            assert!(ordinary_ok && program_ok);
            assert_eq!(ordinary_counts, program_counts, "use_record = {use_record}, not opted in");
            println!("not-opted-in child, use_record = {use_record}: {program_counts:?}");
        }
    }

    #[test]
    fn test_program_signer_circuit_rejects_forgeries() {
        let rng = &mut TestRng::default();
        let signer = sample_vault_signer(rng);
        let (request, input_types) = sample_program(&signer, "vault.aleo", true, true, rng);
        let witness = *request.program_signer().unwrap();
        assert!(verify_in_circuit(&request, &input_types, None, true, "vault.aleo").0);

        // A squatted `tpk` (e.g. copied from another pending transaction) is rejected.
        let squatted = console::Group::rand(rng);
        assert!(!verify_in_circuit(&request, &input_types, Some(squatted), true, "vault.aleo").0);

        // A `tvk` that is not `(tsk * Y).x` is rejected, even with a consistent `tcm`.
        {
            let tvk = console::Field::rand(rng);
            let tcm = <CurrentNetwork as console::Network>::hash_psd2(&[tvk]).unwrap();
            let forged = rebuild(&request, request.input_ids().to_vec(), *request.sk_tag(), tvk, tcm, Some(witness));
            assert!(!verify_in_circuit(&forged, &input_types, None, true, "vault.aleo").0);
        }

        // A second serial number for the same record (a `gamma` that is not `y * H`) is rejected.
        {
            let mut input_ids = request.input_ids().to_vec();
            let console::InputID::Record(commitment, _, record_view_key, _, tag) = input_ids[2] else { panic!() };
            let gamma = console::Group::rand(rng);
            let serial_number =
                console::Record::<CurrentNetwork, console::Plaintext<CurrentNetwork>>::serial_number_from_gamma(
                    &gamma, commitment,
                )
                .unwrap();
            input_ids[2] = console::InputID::Record(commitment, gamma, record_view_key, serial_number, tag);
            let forged = rebuild(&request, input_ids, *request.sk_tag(), *request.tvk(), *request.tcm(), Some(witness));
            assert!(!verify_in_circuit(&forged, &input_types, None, true, "vault.aleo").0);
        }

        // A second tag for the same record (an `sk_tag` not derived from `y`) is rejected.
        {
            let mut input_ids = request.input_ids().to_vec();
            let console::InputID::Record(commitment, gamma, record_view_key, serial_number, _) = input_ids[2] else {
                panic!()
            };
            let sk_tag = console::Field::rand(rng);
            let tag = <CurrentNetwork as console::Network>::hash_psd2(&[sk_tag, commitment]).unwrap();
            input_ids[2] = console::InputID::Record(commitment, gamma, record_view_key, serial_number, tag);
            let forged = rebuild(&request, input_ids, sk_tag, *request.tvk(), *request.tcm(), Some(witness));
            assert!(!verify_in_circuit(&forged, &input_types, None, true, "vault.aleo").0);
        }

        // A wrong `y`, used consistently for `gamma` and `sk_tag`, is rejected (`Y != y * G`).
        {
            let vault = console::ProgramID::from_str("vault.aleo").unwrap();
            let bad = console::ProgramSigner::new_unchecked(
                vault,
                console::PROGRAM_SIGNER_KIND_SIGNER,
                *signer.internal(),
                console::Scalar::rand(rng),
            )
            .unwrap();
            let (forged, input_types) = sample_program(&bad, "vault.aleo", true, true, rng);
            assert!(!verify_in_circuit(&forged, &input_types, None, true, "vault.aleo").0);
        }

        // A root in another opted-in program is rejected (the tweak binds `Y` to `vault.aleo`).
        let (other, input_types) = sample_program(&signer, "vault_two.aleo", true, true, rng);
        assert!(!verify_in_circuit(&other, &input_types, None, true, "vault_two.aleo").0);
        // As a child of another program, the same request is accepted (the root binds `Y`).
        let (child, input_types) = sample_program(&signer, "vault_two.aleo", true, false, rng);
        assert!(verify_in_circuit(&child, &input_types, None, false, "vault_two.aleo").0);

        // A program that has not opted in cannot root a program signer, even for its own address.
        let token = console::ProgramID::from_str("token.aleo").unwrap();
        let token_signer = console::ProgramSigner::from_internal_secret(token, console::Scalar::rand(rng)).unwrap();
        let (token_request, input_types) = sample_program(&token_signer, "token.aleo", true, true, rng);
        assert!(!verify_in_circuit(&token_request, &input_types, None, true, "token.aleo").0);

        // Dropping the program signer witness (claiming the ordinary kind) fails the signature check.
        let flipped =
            rebuild(&request, request.input_ids().to_vec(), *request.sk_tag(), *request.tvk(), *request.tcm(), None);
        assert!(!verify_in_circuit(&flipped, &input_types, None, true, "vault.aleo").0);
    }
}
