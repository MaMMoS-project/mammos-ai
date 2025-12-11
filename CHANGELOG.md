# Changelog

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

This project uses [towncrier](https://towncrier.readthedocs.io/) and the changes for the upcoming release can be found in [changes](changes).

<!-- towncrier release notes start -->

## [mammos-ai 0.2.0](https://github.com/MaMMoS-project/mammos-ai/tree/0.2.0) – 2025-12-11

### Added

- A new AI model which can predict extrinsic magnetic properties (Hc, Mr, BHmax) from the
  intrinsic micromagnetic parameters Ms, A and K has been added. ([#5](https://github.com/MaMMoS-project/mammos-ai/pull/5))
- Added metadata functions. ([#6](https://github.com/MaMMoS-project/mammos-ai/pull/6))

### Misc

- Use [towncrier](https://towncrier.readthedocs.io) to generate changelog from fragments. Each new PR must include a changelog fragment. ([#1](https://github.com/MaMMoS-project/mammos-ai/pull/1))
