# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Entities can name a `label_column`. Results carrying a foreign key to such an entity gain a sibling key with the human-readable label, so answers read as names rather than raw ids. Labels are fetched in one batched, tenant-scoped query per foreign key and can never surface a row the caller could not have read directly.
- Columns marked `hidden` are now stripped from result rows and the model is told never to select them, while remaining usable in `WHERE` clauses. Previously the flag was parsed but had no effect.
- Required filters can bind to a context key other than the column's own name (`param`), pin columns to values fixed in configuration (`constants`), and scope a table through a parent that owns it (`through`). Together these cover polymorphically owned tables and tables with no owner column of their own, neither of which could previously be exposed safely. Chains resolve recursively and fail closed on a missing parent filter or a cycle.

Existing configurations are unaffected: every new key is optional and the previous filter shape behaves exactly as before.

## [0.1.3] - 2026-06-17

### Added

- A generated query that fails validation (for example referencing an unknown table or exceeding the complexity limit) is now sent back to the model along with the validation errors and the list of allowed tables, and regenerated, instead of failing outright. The number of repair attempts is configurable via `validation.repair_attempts` and defaults to 1.

## [0.1.2] - 2026-06-09

### Fixed

- Missing or empty database connection variables now fall back to their defaults instead of producing an invalid connection URL that prevented the schema from loading.
- When the default schema fails to load at startup, requests report the actual underlying reason instead of a generic "no schema loaded" message.
- The version reported by the CLI, the API server, and the health endpoint now derives from the package metadata, so all three stay in sync with the released version.

## [0.1.1] - 2026-06-08

### Security

- Read-only mode now rejects data-modifying statements hidden inside CTEs and subqueries, not just at the top level.
- Per-user data isolation is enforced on the parsed query: a query reading a table with a required filter must constrain its tenant column with a bound parameter, or it is rejected. The filter is also injected automatically and fails closed if it cannot be applied.

### Added

- Role-based bypass for required filters, so trusted roles (e.g. admins) can run system-wide queries while all other callers stay scoped.

## [0.1.0] - 2026-06-07

### Added

- Initial release: natural language to SQL, database first.
