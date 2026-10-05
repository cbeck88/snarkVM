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

/// The payment preference that an address string states for its account.
///
/// The marker exists only in the string. `Address` has no marker, and the `FromStr`, `Parser`, and
/// serde implementations accept only the unmarked form.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum AddressMarker {
    /// The `aleo1` form, which states no preference.
    None,
    /// The `aleox1` form, which marks an exchange that accepts public transfers only.
    Exchange,
}

/// The prefix of an exchange address, which is the unmarked prefix `aleo1` with `x` inserted at index 4.
const EXCHANGE_PREFIX: &str = "aleox1";

/// The index of the exchange marker in an exchange address.
const MARKER_INDEX: usize = 4;

impl<E: Environment> Address<E> {
    /// Returns the address string with the given marker.
    ///
    /// The marked form copies the unmarked form, including its checksum, and inserts the marker.
    pub fn to_marked_string(&self, marker: AddressMarker) -> String {
        let mut string = self.to_string();
        if marker == AddressMarker::Exchange {
            string.insert(MARKER_INDEX, 'x');
        }
        string
    }

    /// Decodes an address string in either the unmarked or the marked form.
    ///
    /// The checksum of a marked string covers the unmarked form, so it does not cover the marker.
    pub fn from_marked_str(string: &str) -> Result<(Self, AddressMarker)> {
        match string.len() {
            63 => Ok((Self::from_str(string)?, AddressMarker::None)),
            64 => {
                ensure!(string.starts_with(EXCHANGE_PREFIX), "Invalid exchange address: expected the prefix 'aleox1'");
                // The prefix is ASCII, so `MARKER_INDEX` and `MARKER_INDEX + 1` are character boundaries.
                let unmarked = [&string[..MARKER_INDEX], &string[MARKER_INDEX + 1..]].concat();
                Ok((Self::from_str(&unmarked)?, AddressMarker::Exchange))
            }
            length => bail!("Invalid account address length: found {length}, expected 63 or 64"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use snarkvm_console_network_environment::Console;

    type CurrentEnvironment = Console;

    const ITERATIONS: u64 = 1_000;

    /// The account that every test vector belongs to.
    const ADDRESS: &str = "aleo1ml2xr6fawppd6uaf8gn95uy2fpqqg8gk74k0lu8na7uvayk64v8qu8hw5u";
    const EXCHANGE_ADDRESS: &str = "aleox1ml2xr6fawppd6uaf8gn95uy2fpqqg8gk74k0lu8na7uvayk64v8qu8hw5u";

    #[test]
    fn test_accepted_vectors() -> Result<()> {
        let (unmarked, marker) = Address::<CurrentEnvironment>::from_marked_str(ADDRESS)?;
        assert_eq!(AddressMarker::None, marker);
        assert_eq!(ADDRESS, unmarked.to_string());

        let (marked, marker) = Address::<CurrentEnvironment>::from_marked_str(EXCHANGE_ADDRESS)?;
        assert_eq!(AddressMarker::Exchange, marker);
        assert_eq!(unmarked, marked);
        assert_eq!(EXCHANGE_ADDRESS, marked.to_marked_string(AddressMarker::Exchange));
        Ok(())
    }

    #[test]
    fn test_rejected_vectors() {
        for string in [
            // The checksum was recomputed over `aleox`.
            "aleox1ml2xr6fawppd6uaf8gn95uy2fpqqg8gk74k0lu8na7uvayk64v8q7y7vj3",
            // The checksum fails.
            "aleox1ml2xr6fawppd6uaf8gn95uy2fpqqg8gk74k0lu8na7uvayk64v8qu8hw5q",
            // The address is 63 characters and does not start with `aleo1`.
            "aleox1ml2xr6fawppd6uaf8gn95uy2fpqqg8gk74k0lu8na7uvayk64v8qu8hw5",
            // The address is uppercase.
            "ALEOX1ML2XR6FAWPPD6UAF8GN95UY2FPQQG8GK74K0LU8NA7UVAYK64V8QU8HW5U",
            // The unmarked address is uppercase.
            "ALEO1ML2XR6FAWPPD6UAF8GN95UY2FPQQG8GK74K0LU8NA7UVAYK64V8QU8HW5U",
            // The address is 64 characters and starts with `aleo1`.
            "aleo1ml_2xr6fawppd6uaf8gn95uy2fpqqg8gk74k0lu8na7uvayk64v8qu8hw5u",
            // The address contains a `_` separator.
            "aleo1ml2xr6fawppd6uaf8gn95uy2fpqqg8gk74k0lu8na7uvayk64v8qu8hw_5",
            // The address is 64 characters with a marker other than `x`.
            "aleoq1ml2xr6fawppd6uaf8gn95uy2fpqqg8gk74k0lu8na7uvayk64v8qu8hw5u",
            "",
        ] {
            assert!(Address::<CurrentEnvironment>::from_marked_str(string).is_err(), "accepted {string}");
        }
    }

    #[test]
    fn test_unmarked_parsers_reject_exchange_address() {
        assert!(Address::<CurrentEnvironment>::from_str(EXCHANGE_ADDRESS).is_err());
        assert!(Address::<CurrentEnvironment>::parse(EXCHANGE_ADDRESS).is_err());
        assert!(serde_json::from_str::<Address<CurrentEnvironment>>(&format!("\"{EXCHANGE_ADDRESS}\"")).is_err());
    }

    #[test]
    fn test_round_trip() -> Result<()> {
        let mut rng = TestRng::default();

        for _ in 0..ITERATIONS {
            let expected = Address::<CurrentEnvironment>::rand(&mut rng);
            for marker in [AddressMarker::None, AddressMarker::Exchange] {
                let string = expected.to_marked_string(marker);
                assert_eq!((expected, marker), Address::from_marked_str(&string)?);
            }
            // The marked form differs from the unmarked form only by the marker.
            let unmarked = expected.to_marked_string(AddressMarker::None);
            let marked = expected.to_marked_string(AddressMarker::Exchange);
            assert_eq!(unmarked, marked.replacen("aleox1", "aleo1", 1));
        }
        Ok(())
    }
}
