# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### 📚 Documentation

- ct-conventions pinned at `v0.7.0` in `AGENTS.md`: the release rules `REL1`–`REL7` in the
  index, and a *Recurring tasks* section pointing to the release recipe of ct-agents.

### 🧪 Testing

- Release workflows: `ReleaseGate.yml` (the gate of a release PR, `REL1`–`REL7`) and
  `ReleaseTag.yml` (tag and GitHub release of a merged release PR) call CTActions;
  `Breakage.yml` moves to `breakage-suite.yml` (one report, the `REL1` check of a release
  PR) and also runs on push to `main`; `Documentation.yml` accepts `workflow_dispatch`.

