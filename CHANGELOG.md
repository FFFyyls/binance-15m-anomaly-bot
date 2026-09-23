# Changelog

All notable changes to this project will be documented in this file.

The format is based on Keep a Changelog, and the project follows Semantic
Versioning once stable releases begin.

## [Unreleased]

### Changed

- standardized the repository to an English `src`-layout Python project;
- separated source, backtests, scripts, configuration, local data, and output;
- made dry-run and testnet the example and local safety defaults;
- preserved runtime databases unless reset is explicitly requested;
- added a production-order confirmation guard for the manual stop-order tool;
- added Apache License 2.0, tests, packaging metadata, and CI.
