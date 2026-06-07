# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.1] - 2026-06-08

### Security

- Read-only mode now rejects data-modifying statements hidden inside CTEs and subqueries, not just at the top level.
- Per-user data isolation is enforced on the parsed query: a query reading a table with a required filter must constrain its tenant column with a bound parameter, or it is rejected. The filter is also injected automatically and fails closed if it cannot be applied.

### Added

- Role-based bypass for required filters, so trusted roles (e.g. admins) can run system-wide queries while all other callers stay scoped.

## [0.1.0] - 2026-06-07

### Added

- Initial release: natural language to SQL, database first.
