# duplication-mcp

Finds duplicated code and ranks it by what it costs to keep. For AI agents, over MCP.

## Why not the alternatives

Other tools match structure exactly, so a copy with one renamed variable is
invisible — while every accessor and guard clause gets reported as loudly as a
real problem. The output is noise, so nobody reads it.

This one:

- **Finds copies that drifted** — matches by meaning, so renamed copies still
  count. Even catches the same logic rewritten in another language.
- **Ranks by cost** — a 60-line block copied 4× beats a 3-line one repeated 40×.
- **Ignores idiom** — boilerplate is demoted, with the reason, not dumped on you.

### Languages

C#, TypeScript, JavaScript, Java, Kotlin, Go, Rust, Ruby, PHP, Python, C, C++,
Objective-C, Swift, Scala, VB, SQL, shell, PowerShell, CSS/SCSS/Less.

There is no parser or grammar to install — block boundaries are inferred from
braces and indentation, so anything brace- or indent-structured works. The
trade-off is that reported line ranges are approximate; read the returned source
for the exact extent.

## Why give an agent this

- **Every finding has a confidence** (`certain` → `low`), so the agent knows what
  it can act on and what it must read first.
- **Answers say how complete they are** — files still being indexed are counted
  in every reply, so a partial answer is never mistaken for a full one.
- **Findings are traceable** — `explain_duplication` returns the source of every
  copy.

| Tool | Answers |
|---|---|
| `detect_duplication` | Where is the duplication, worst first? |
| `duplication_status` | Is the index ready? |
| `explain_duplication` | Show me every copy. |
| `reindex` | Start over. |

## Install

Node 22.5+.

```bash
npm install && npm run build
node dist/index.js download-model      # ~154 MB, once
```

Then add it to `.mcp.json` in your project root:

```json
{
  "mcpServers": {
    "duplication": {
      "command": "node",
      "args": ["/path/to/duplication/dist/index.js", "."]
    }
  }
}
```

It indexes on startup and keeps up with edits. Without the model it still runs —
it just reports nothing and tells you to install it.

### Scoping it

Optional `duplication.config.json` in your project root. Use `include` as a
whitelist — name the paths to analyse and **everything else is ignored**:

```json
{
  "include": ["src/**", "lib/**"],
  "exclude": ["**/*.generated.*", "**/migrations/**"],
  "minLines": 6,
  "similarityThreshold": 0.45
}
```

`include` defaults to `["**"]` (everything). `exclude` wins over `include`, so
you can whitelist a tree and still drop generated files inside it. Other
defaults are calibrated and worth leaving alone.

Git submodules and vendored clones are left out: they are other projects, and
duplication inside them cannot be fixed from here. Set
`"includeNestedRepositories": true` if you own them too.

Edits take effect on the next question, so an agent can write this file itself.
Until it exists, every source file under the root is in scope — which is what
the replies say, along with the largest folders, when the project is big enough
for that to matter.

---

[How it works](docs/design.md)
