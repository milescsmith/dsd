## [0.6.1] - 2024-09-16

### Changed

- Make scVI and scAR optional installs

## [0.6.0] - 2024-09-16

### Changed

- Fixed `dsd.scrnaseq.std_quality_contol` so that it ACTUALLY FILTERS OUT CELLS WITH HIGH MITO READS!

## [0.5.0] - 2024-09-16

### Added

- A verson of `dsd.scrnaseq.std_quality_contol` that functions on an anndata object

## [0.4.0] - 2024-09-01

### Added

- Added `std_quality_control()` and `std_process()`

## [0.3.1] - 2024-08-14

### Fixed

- Fixed a few stray typos

## [0.3.0] - 2024-08-14

### Added
- Ability to specify the sample folder instead of the actual sample matrices

### Changed
- Split code into separate files
- Catch more warnings and ignore the ones that consistently pop up and make no difference

## [0.2.0] - 2024-08-14

### Fixed

- Fixed double logging
- Fixed getting the version of the package


## [0.1.0] - 2024-08-13

## Added

- All

[0.6.0]: https://github.com/milescsmith/dsd/releases/compare/0.5.0..0.6.0
[0.5.0]: https://github.com/milescsmith/dsd/releases/compare/0.4.0..0.5.0
[0.4.0]: https://github.com/milescsmith/dsd/releases/compare/0.3.1..0.4.0
[0.3.1]: https://github.com/milescsmith/dsd/releases/compare/0.3.0..0.3.1
[0.3.0]: https://github.com/milescsmith/dsd/releases/compare/0.2.0..0.3.0
[0.2.0]: https://github.com/milescsmith/dsd/releases/compare/0.1.0..0.2.0
[0.1.0]: https://github.com/milescsmith/dsd/releases/tag/v0.1.0