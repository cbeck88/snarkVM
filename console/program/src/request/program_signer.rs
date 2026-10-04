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

//! Program-owned signers (exploratory prototype).
//!
//! A program `P` owns the address `Y = X + Ht(kind, P, X)·G`, where `X = x·G` is an internal key
//! chosen by the program's user and `Ht` is the tweak hash below. The view key of `Y` is
//! `y = x + Ht(kind, P, X)`. An execution rooted in `P` may use `Y` as its signer without a
//! signature: the root transition proves the tweak against its own program, and `P`'s code is the
//! authorization.

use super::*;

/// The kind of a program-owned address that acts as the signer of executions rooted in its program.
pub const PROGRAM_SIGNER_KIND_SIGNER: u8 = 1;

/// The domain separator for the program signer tweak hash.
pub const PROGRAM_SIGNER_TWEAK_DOMAIN: &str = "AleoProgramSignerTweak0";

/// Returns the domain separator for the program signer tweak hash, as a field element.
pub fn program_signer_tweak_domain<N: Network>() -> Field<N> {
    Field::new_domain_separator(PROGRAM_SIGNER_TWEAK_DOMAIN)
}

/// Returns the tweak preimage `[domain, kind, P.x, X.x, X.y]` as a plaintext array of fields.
///
/// The preimage is a plaintext so that a program can recompute the tweak with
/// `hash.psd4 <[field; 5u32]> into <r> as scalar`.
pub fn program_signer_tweak_preimage<N: Network>(
    kind: u8,
    program_address: &Address<N>,
    internal: &Group<N>,
) -> Plaintext<N> {
    let elements = [
        program_signer_tweak_domain::<N>(),
        Field::from_u8(kind),
        program_address.to_x_coordinate(),
        internal.to_x_coordinate(),
        internal.to_y_coordinate(),
    ];
    Plaintext::Array(elements.into_iter().map(|f| Plaintext::from(Literal::Field(f))).collect(), Default::default())
}

/// Returns the tweak `Ht(kind, P, X) = HashToScalarPSD4([domain, kind, P.x, X.x, X.y])`.
pub fn program_signer_tweak<N: Network>(
    kind: u8,
    program_address: &Address<N>,
    internal: &Group<N>,
) -> Result<Scalar<N>> {
    N::hash_to_scalar_psd4(&program_signer_tweak_preimage(kind, program_address, internal).to_fields()?)
}

/// Returns the program-owned address `Y = X + Ht(kind, P, X)·G`.
pub fn program_signer_address<N: Network>(
    kind: u8,
    program_address: &Address<N>,
    internal: &Group<N>,
) -> Result<Address<N>> {
    let tweak = program_signer_tweak(kind, program_address, internal)?;
    Ok(Address::new(*internal + N::g_scalar_multiply(&tweak)))
}

/// Returns `true` if the given program has opted in to rooting program-signer executions.
///
/// TODO(prototype): this is a stand-in for a program-level opt-in flag in the bytecode format.
/// A program opts in if its name starts with `vault`.
pub fn program_signer_opt_in<N: Network>(program_id: &ProgramID<N>) -> bool {
    program_id.name().to_string().starts_with("vault")
}

/// The private witness that a request carries when its signer is a program-owned address.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct ProgramSignerWitness<N: Network> {
    /// The kind of the program-owned address.
    kind: u8,
    /// The internal key `X`.
    internal: Group<N>,
    /// The view key `y` of the program-owned address `Y`.
    view_key: Scalar<N>,
    /// The transition secret key `tsk`, with `tpk = tsk·G` and `tvk = (tsk·Y).x`.
    tsk: Scalar<N>,
}

impl<N: Network> ProgramSignerWitness<N> {
    /// Initializes a new program signer witness.
    pub const fn new(kind: u8, internal: Group<N>, view_key: Scalar<N>, tsk: Scalar<N>) -> Self {
        Self { kind, internal, view_key, tsk }
    }

    /// Returns the kind of the program-owned address.
    pub const fn kind(&self) -> u8 {
        self.kind
    }

    /// Returns the internal key `X`.
    pub const fn internal(&self) -> &Group<N> {
        &self.internal
    }

    /// Returns the view key `y`.
    pub const fn view_key(&self) -> &Scalar<N> {
        &self.view_key
    }

    /// Returns the transition secret key `tsk`.
    pub const fn tsk(&self) -> &Scalar<N> {
        &self.tsk
    }
}

impl<N: Network> FromBytes for ProgramSignerWitness<N> {
    fn read_le<R: Read>(mut reader: R) -> IoResult<Self> {
        let kind = u8::read_le(&mut reader)?;
        let internal = FromBytes::read_le(&mut reader)?;
        let view_key = FromBytes::read_le(&mut reader)?;
        let tsk = FromBytes::read_le(&mut reader)?;
        Ok(Self { kind, internal, view_key, tsk })
    }
}

impl<N: Network> ToBytes for ProgramSignerWitness<N> {
    fn write_le<W: Write>(&self, mut writer: W) -> IoResult<()> {
        self.kind.write_le(&mut writer)?;
        self.internal.write_le(&mut writer)?;
        self.view_key.write_le(&mut writer)?;
        self.tsk.write_le(&mut writer)
    }
}

/// A program-owned signer: the program `P`, the internal key `X`, and the view key `y` of
/// `Y = X + Ht(kind, P, X)·G`.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct ProgramSigner<N: Network> {
    /// The program that owns the address.
    program_id: ProgramID<N>,
    /// The kind of the program-owned address.
    kind: u8,
    /// The internal key `X`.
    internal: Group<N>,
    /// The view key `y`.
    view_key: Scalar<N>,
    /// The program-owned address `Y`.
    address: Address<N>,
}

impl<N: Network> ProgramSigner<N> {
    /// Initializes a program signer of kind `signer` from the internal secret `x`.
    pub fn from_internal_secret(program_id: ProgramID<N>, x: Scalar<N>) -> Result<Self> {
        let kind = PROGRAM_SIGNER_KIND_SIGNER;
        let internal = N::g_scalar_multiply(&x);
        let tweak = program_signer_tweak(kind, &program_id.to_address()?, &internal)?;
        Self::new(program_id, kind, internal, x + tweak)
    }

    /// Initializes a program signer from the internal key `X` and the view key `y`.
    /// Fails if `y·G != X + Ht(kind, P, X)·G`.
    pub fn new(program_id: ProgramID<N>, kind: u8, internal: Group<N>, view_key: Scalar<N>) -> Result<Self> {
        ensure!(kind == PROGRAM_SIGNER_KIND_SIGNER, "Unsupported program signer kind '{kind}'");
        let address = program_signer_address(kind, &program_id.to_address()?, &internal)?;
        ensure!(*address == N::g_scalar_multiply(&view_key), "The view key does not match the program-owned address");
        Ok(Self { program_id, kind, internal, view_key, address })
    }

    /// Initializes a program signer without checking that the view key matches.
    /// This exists to construct invalid signers in negative tests.
    #[doc(hidden)]
    pub fn new_unchecked(program_id: ProgramID<N>, kind: u8, internal: Group<N>, view_key: Scalar<N>) -> Result<Self> {
        let address = program_signer_address(kind, &program_id.to_address()?, &internal)?;
        Ok(Self { program_id, kind, internal, view_key, address })
    }

    /// Initializes a program signer with an arbitrary address and no checks.
    /// This exists to construct invalid signers in negative tests.
    #[doc(hidden)]
    pub fn from_parts_unchecked(
        program_id: ProgramID<N>,
        kind: u8,
        internal: Group<N>,
        view_key: Scalar<N>,
        address: Address<N>,
    ) -> Self {
        Self { program_id, kind, internal, view_key, address }
    }

    /// Returns the program that owns the address.
    pub const fn program_id(&self) -> &ProgramID<N> {
        &self.program_id
    }

    /// Returns the kind of the program-owned address.
    pub const fn kind(&self) -> u8 {
        self.kind
    }

    /// Returns the internal key `X`.
    pub const fn internal(&self) -> &Group<N> {
        &self.internal
    }

    /// Returns the view key `y`, which decrypts records owned by `Y`.
    pub fn view_key(&self) -> ViewKey<N> {
        ViewKey::from_scalar(self.view_key)
    }

    /// Returns the program-owned address `Y`.
    pub const fn address(&self) -> &Address<N> {
        &self.address
    }
}

/// The signer of a request: an account private key, or a program-owned address.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum RequestSigner<N: Network> {
    /// An ordinary account, which signs each request.
    Account(PrivateKey<N>),
    /// A program-owned address, which signs nothing.
    Program(ProgramSigner<N>),
}

impl<N: Network> From<PrivateKey<N>> for RequestSigner<N> {
    fn from(private_key: PrivateKey<N>) -> Self {
        Self::Account(private_key)
    }
}

impl<N: Network> From<ProgramSigner<N>> for RequestSigner<N> {
    fn from(program_signer: ProgramSigner<N>) -> Self {
        Self::Program(program_signer)
    }
}

impl<N: Network> RequestSigner<N> {
    /// Returns the address of the signer.
    pub fn address(&self) -> Result<Address<N>> {
        match self {
            Self::Account(private_key) => Address::try_from(private_key),
            Self::Program(program_signer) => Ok(*program_signer.address()),
        }
    }

    /// Returns the private key, if the signer is an account.
    pub const fn private_key(&self) -> Option<&PrivateKey<N>> {
        match self {
            Self::Account(private_key) => Some(private_key),
            Self::Program(..) => None,
        }
    }

    /// Returns a request for the given program function and inputs. See `Request::sign`.
    #[allow(clippy::too_many_arguments)]
    pub fn sign<R: Rng + CryptoRng>(
        &self,
        program_id: ProgramID<N>,
        function_name: Identifier<N>,
        inputs: impl ExactSizeIterator<Item = impl TryInto<Value<N>>>,
        input_types: &[ValueType<N>],
        root_tvk: Option<Field<N>>,
        is_root: bool,
        program_checksum: Option<Field<N>>,
        is_dynamic: bool,
        rng: &mut R,
    ) -> Result<Request<N>> {
        match self {
            Self::Account(private_key) => Request::sign(
                private_key,
                program_id,
                function_name,
                inputs,
                input_types,
                root_tvk,
                is_root,
                program_checksum,
                is_dynamic,
                rng,
            ),
            Self::Program(program_signer) => Request::sign_as_program(
                program_signer,
                program_id,
                function_name,
                inputs,
                input_types,
                root_tvk,
                is_root,
                program_checksum,
                is_dynamic,
                rng,
            ),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use snarkvm_console_network::MainnetV0;

    type CurrentNetwork = MainnetV0;

    fn sample_request(
        program_signer: &ProgramSigner<CurrentNetwork>,
        program_id: &str,
        is_root: bool,
        rng: &mut TestRng,
    ) -> (Request<CurrentNetwork>, Vec<ValueType<CurrentNetwork>>) {
        let address = program_signer.address();
        let record = format!(
            "{{ owner: {address}.private, token_amount: 7u64.private, _nonce: 2293253577170800572742339369209137467208538700597121244293392265726446806023group.public }}"
        );
        let inputs = vec![Value::from_str(&record).unwrap(), Value::from_str("5u64").unwrap()];
        let input_types =
            vec![ValueType::from_str("token.record").unwrap(), ValueType::from_str("u64.private").unwrap()];
        let request = Request::sign_as_program(
            program_signer,
            ProgramID::from_str(program_id).unwrap(),
            Identifier::from_str("transfer").unwrap(),
            inputs.into_iter(),
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

    #[test]
    fn test_program_signer_address() {
        let rng = &mut TestRng::default();
        let program_id = ProgramID::<CurrentNetwork>::from_str("vault.aleo").unwrap();
        let x = Scalar::rand(rng);
        let signer = ProgramSigner::from_internal_secret(program_id, x).unwrap();
        // `Y = y * G`, and `y = x + Ht(kind, P, X)`.
        assert_eq!(**signer.address(), CurrentNetwork::g_scalar_multiply(&signer.view_key()));
        let tweak = program_signer_tweak(
            PROGRAM_SIGNER_KIND_SIGNER,
            &program_id.to_address().unwrap(),
            &CurrentNetwork::g_scalar_multiply(&x),
        )
        .unwrap();
        assert_eq!(*signer.view_key(), x + tweak);
        // The view key derives the address `Y`.
        assert_eq!(Address::try_from(signer.view_key()).unwrap(), *signer.address());
        // A wrong view key is rejected.
        assert!(
            ProgramSigner::new(program_id, PROGRAM_SIGNER_KIND_SIGNER, *signer.internal(), Scalar::rand(rng)).is_err()
        );
    }

    #[test]
    fn test_program_signer_request_verify() {
        let rng = &mut TestRng::default();
        let vault = ProgramID::<CurrentNetwork>::from_str("vault.aleo").unwrap();
        let signer = ProgramSigner::from_internal_secret(vault, Scalar::rand(rng)).unwrap();

        // At the root of its own (opted-in) program, the request verifies.
        let (request, input_types) = sample_request(&signer, "vault.aleo", true, rng);
        assert!(request.verify(&input_types, true, None));
        // `tpk` is `tsk * G`, and `tvk` is `(tsk * Y).x`.
        let tsk = *request.program_signer().unwrap().tsk();
        assert_eq!(request.to_tpk(), CurrentNetwork::g_scalar_multiply(&tsk));
        // As a child of any program, the request verifies (the root binds `Y`).
        let (request, input_types) = sample_request(&signer, "token.aleo", false, rng);
        assert!(request.verify(&input_types, false, None));
        // At the root of another program, the request fails.
        let (request, input_types) = sample_request(&signer, "token.aleo", true, rng);
        assert!(!request.verify(&input_types, true, None));
        // At the root of another opted-in program, the request fails.
        let (request, input_types) = sample_request(&signer, "vault_two.aleo", true, rng);
        assert!(!request.verify(&input_types, true, None));
        // A non-opted-in program cannot root a program signer, even for its own address.
        let token = ProgramID::<CurrentNetwork>::from_str("token.aleo").unwrap();
        let token_signer = ProgramSigner::from_internal_secret(token, Scalar::rand(rng)).unwrap();
        let (request, input_types) = sample_request(&token_signer, "token.aleo", true, rng);
        assert!(!request.verify(&input_types, true, None));
        // A wrong view key fails.
        let bad =
            ProgramSigner::new_unchecked(vault, PROGRAM_SIGNER_KIND_SIGNER, *signer.internal(), Scalar::rand(rng))
                .unwrap();
        let (request, input_types) = sample_request(&bad, "vault.aleo", true, rng);
        assert!(!request.verify(&input_types, true, None));
    }

    #[test]
    fn test_program_signer_request_bytes() {
        let rng = &mut TestRng::default();
        let vault = ProgramID::<CurrentNetwork>::from_str("vault.aleo").unwrap();
        let signer = ProgramSigner::from_internal_secret(vault, Scalar::rand(rng)).unwrap();
        let (request, _) = sample_request(&signer, "vault.aleo", true, rng);
        let bytes = request.to_bytes_le().unwrap();
        assert_eq!(bytes[0], 3);
        assert_eq!(request, Request::read_le(&bytes[..]).unwrap());
    }
}
