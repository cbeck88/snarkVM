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

//! Checks that reading a program does not allocate in proportion to a declared element count.
//!
//! `MAX_COMMANDS` and `MAX_INSTRUCTIONS` are both `u16::MAX`, and `Command` and `Instruction` are
//! several hundred bytes wide, so honouring a declared count up front would let a few dozen bytes
//! of input ask the allocator for tens of megabytes. This lives in its own integration test
//! because it installs a global allocator, which would otherwise affect every other test in the
//! binary.

use snarkvm_console::{network::MainnetV0, program::ProgramID};
use snarkvm_synthesizer_program::Program;
use snarkvm_utilities::{FromBytes, ToBytes};

use std::{
    alloc::{GlobalAlloc, Layout, System},
    str::FromStr,
    sync::atomic::{AtomicUsize, Ordering},
};

type CurrentNetwork = MainnetV0;

static LIVE: AtomicUsize = AtomicUsize::new(0);
static PEAK: AtomicUsize = AtomicUsize::new(0);

/// Tracks the high-water mark of live allocated bytes.
///
/// Note this counts bytes *requested*, which is the figure that matters here: a large
/// `Vec::with_capacity` is mapped rather than faulted in, so resident memory would not show it.
struct PeakTracking;

unsafe impl GlobalAlloc for PeakTracking {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let live = LIVE.fetch_add(layout.size(), Ordering::Relaxed) + layout.size();
        PEAK.fetch_max(live, Ordering::Relaxed);
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        LIVE.fetch_sub(layout.size(), Ordering::Relaxed);
        unsafe { System.dealloc(ptr, layout) }
    }
}

#[global_allocator]
static ALLOCATOR: PeakTracking = PeakTracking;

/// Builds the shortest program that declares `num_commands` constructor commands and then ends.
fn program_declaring_commands(num_commands: u16) -> Vec<u8> {
    let mut bytes = vec![1u8]; // Version.
    bytes.extend(ProgramID::<CurrentNetwork>::from_str("test.aleo").unwrap().to_bytes_le().unwrap());
    bytes.push(0u8); // No imports.
    bytes.extend(1u16.to_le_bytes()); // One component.
    bytes.push(5u8); // The constructor variant.
    bytes.extend(num_commands.to_le_bytes());
    // Nothing follows, so the read fails as soon as it asks for the first command.
    bytes
}

#[test]
fn reading_a_program_does_not_allocate_for_a_declared_count() {
    // Warm anything lazily initialized before the measurement starts.
    let _ = Program::<CurrentNetwork>::from_bytes_le(&program_declaring_commands(1));

    let bytes = program_declaring_commands(u16::MAX);
    assert!(bytes.len() < 32, "the declaration should be tiny, got {} bytes", bytes.len());

    let before = LIVE.load(Ordering::Relaxed);
    PEAK.store(before, Ordering::Relaxed);

    // The program is truncated, so this must fail -- the point is what it allocates on the way.
    assert!(Program::<CurrentNetwork>::from_bytes_le(&bytes).is_err());

    let peak = PEAK.load(Ordering::Relaxed).saturating_sub(before);

    // Honouring the declared count would ask for 65,535 * size_of::<Command>() = 54 MiB. The
    // reserve is `MAX_EAGER_RESERVE` elements, so the real figure is under a megabyte; leave room for
    // whatever else the read touches rather than pinning an exact number.
    const LIMIT: usize = 4 * 1024 * 1024;
    assert!(peak < LIMIT, "reading a {}-byte program allocated {peak} bytes", bytes.len());
}
