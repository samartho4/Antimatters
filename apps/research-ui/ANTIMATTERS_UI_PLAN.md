# Antimatters UI Redesign Plan

## Vision: Information → Computation → Evolution

Based on the physics metaphor where:
- **Information**: What goes IN (tools, context, data sources)
- **Computation**: The PROCESS (agent execution, conversations)
- **Evolution**: What comes OUT (artifacts, learnings, feedback)

---

## Current vs Target Architecture

### Current Layout
```
┌──────────────────────────────────────────────────────────┐
│ CopilotSidebar (right, wraps everything)                 │
│ ┌──────────────────────────────────────────────────────┐ │
│ │ Header: Logo | Artifacts btn | Status               │ │
│ ├─────────────────────┬────────────────────────────────┤ │
│ │ Chat (50%)          │ ExperimentRenderer (50%)       │ │
│ │ - Messages          │ - Protocol                     │ │
│ │ - TraceStream       │ - Materials                    │ │
│ │ - Input             │ - Results                      │ │
│ └─────────────────────┴────────────────────────────────┘ │
│ ArtifactDrawer (sliding right)                           │
└──────────────────────────────────────────────────────────┘
```

### Target Layout (Antigravity-inspired)
```
┌────────────────────────────────────────────────────────────────────┐
│                        Agent Manager                    Settings   │
├──────────────┬─────────────────────────────────────────────────────┤
│   SIDEBAR    │              COMPUTATION                            │
│   (240px)    │                                                     │
│              │  ┌───────────────────────────────────────────────┐  │
│ + New Chat   │  │ Start new simulation in  ▼ core               │  │
│              │  │                                                │  │
│ WORKSPACES   │  │ ┌─────────────────────────────────────────┐   │  │
│ ▸ core    +  │  │ │ Ask anything, @ for context             │   │  │
│ + Open       │  │ │ [+] [Mode ▼] [Model ▼]      [🎤] [→]   │   │  │
│              │  │ └─────────────────────────────────────────┘   │  │
│ ─────────────│  │                                                │  │
│ INFORMATION  │  │  ┌────────────────────────────────────────┐   │  │
│ 🔧 Tools     │  │  │ [Artifacts render inline as cards]     │   │  │
│ 📁 Context   │  │  │                                        │   │  │
│ 📚 Literature│  │  │ TraceStream / Messages                 │   │  │
│              │  │  │                                        │   │  │
│ ─────────────│  │  └────────────────────────────────────────┘   │  │
│ EVOLUTION    │  │                                                │  │
│ 📊 Artifacts │  └───────────────────────────────────────────────┘  │
│ 💬 Feedback  │                                                     │
│              │  (Optional: Right panel for 3D viewer / details)    │
│ ─────────────│                                                     │
│ ⚙ Settings   │                                                     │
│ 💡 Feedback  │                                                     │
└──────────────┴─────────────────────────────────────────────────────┘
```

---

## Component Mapping

| Antigravity UI Element | Antimatters Equivalent | Implementation |
|------------------------|------------------------|----------------|
| Inbox | Recent Chats | List of conversation threads |
| Start conversation | + New Chat | Creates new simulation thread |
| Workspaces > core | Workspaces > core | Agent/project selector |
| Open Workspace | + Open | Load saved workspace |
| Playground → **Information** | Tools, Context, Literature | MCP tool browser, data sources |
| Knowledge → **Evolution** | Artifacts, Feedback | Artifact gallery, feedback history |
| Main chat area → **Computation** | Simulation workspace | Chat + inline artifacts |
| ~~Open editor~~ | REMOVED | - |
| ~~Use Playground~~ | REMOVED | - |

---

## Implementation Phases

### Phase 1: Custom Sidebar Component
Replace CopilotSidebar wrapper with custom layout.

**New file: `components/Sidebar.tsx`**
```tsx
// Sections:
// 1. Logo + New Chat button
// 2. Workspaces (collapsible list of projects/agents)
// 3. Information (Tools, Context, Literature)
// 4. Evolution (Artifacts, Feedback)
// 5. Footer (Settings, Help)
```

**Key features:**
- Collapsible sections with smooth animations
- Workspace switcher (currently just "core")
- Tool browser showing available MCP tools
- Artifact gallery with filtering
- Feedback history viewer

### Phase 2: Main Computation Area
Refactor the chat/workspace area.

**Changes to `App.tsx`:**
- Remove CopilotSidebar wrapper
- Use CopilotKit headless with `useCopilotChat`
- Custom chat input with mode/model selectors
- Inline artifact cards in message stream

**Input enhancements:**
- `@ for context` - mention artifacts/tools
- Mode selector: Planning / Execution / Analysis
- Model selector: Gemini 2.0 Flash / Pro

### Phase 3: Evolution Panel (Artifacts)
Transform ArtifactDrawer into sidebar section + optional detail panel.

**Features:**
- Sidebar: Compact artifact list with icons
- Click: Opens detail panel OR loads into main view
- Feedback: Comments/annotations on artifacts
- History: Track artifact evolution over time

### Phase 4: AG-UI Event Integration
Leverage AG-UI protocol for real-time updates.

**Event handling:**
```tsx
// ACTIVITY_SNAPSHOT → Task list artifact
// ACTIVITY_DELTA → Progress updates
// CUSTOM:artifact_created → New artifact card
// STATE_DELTA → Workspace state sync
// TOOL_CALL_* → Tool execution UI
```

### Phase 5: CopilotKit Integration
Keep CopilotKit for its powerful features but use headless mode.

**Hooks to use:**
- `useCopilotChat` - Core chat functionality
- `useCopilotReadable` - Share state with AI (already using)
- `useCopilotAction` - Frontend tools with Generative UI
- `useFrontendTool` - Tool execution with custom render

**Generative UI for tools:**
```tsx
useCopilotAction({
  name: "fetch_ped_ensemble",
  available: "disabled", // Render only
  render: ({ status, args, result }) => {
    if (status === "inProgress") return <LoadingCard tool="PED" />;
    if (status === "complete") return <StructureCard data={result} />;
  }
});
```

---

## File Structure Changes

```
src/
├── App.tsx                    # Main layout with sidebar
├── components/
│   ├── Sidebar/
│   │   ├── index.tsx          # Main sidebar component
│   │   ├── WorkspaceList.tsx  # Workspace navigation
│   │   ├── InformationPanel.tsx # Tools, Context, Literature
│   │   └── EvolutionPanel.tsx # Artifacts, Feedback
│   ├── Computation/
│   │   ├── index.tsx          # Main workspace area
│   │   ├── ChatInput.tsx      # Enhanced input with selectors
│   │   ├── MessageStream.tsx  # Messages + inline artifacts
│   │   └── TraceStream.tsx    # (existing) execution trace
│   ├── Artifacts/
│   │   ├── ArtifactCard.tsx   # Inline artifact display
│   │   ├── ArtifactDetail.tsx # Full artifact view
│   │   └── ArtifactRenderer.tsx # (existing) type-specific renders
│   └── shared/
│       ├── Header.tsx         # Top bar: Agent Manager + Settings
│       └── StatusIndicator.tsx
├── services/
│   ├── aguiService.ts         # (existing) AG-UI protocol
│   └── workspaceService.ts    # Workspace/project management
└── types.ts
```

---

## Design Tokens

```css
/* Antimatters Dark Theme (matching Antigravity) */
--am-bg-primary: #1a1d21;      /* Main background */
--am-bg-secondary: #242830;    /* Sidebar, cards */
--am-bg-tertiary: #2d323c;     /* Hover states */
--am-text-primary: #e4e7eb;    /* Primary text */
--am-text-secondary: #8b929e;  /* Secondary text */
--am-text-muted: #5c6370;      /* Muted text */
--am-accent: #6366f1;          /* Primary accent (indigo) */
--am-accent-hover: #818cf8;    /* Accent hover */
--am-border: #323842;          /* Borders */
--am-success: #22c55e;         /* Success states */
--am-warning: #eab308;         /* Warning states */
--am-error: #ef4444;           /* Error states */

/* Sidebar widths */
--sidebar-width: 240px;
--sidebar-collapsed: 64px;
```

---

## Key UX Decisions

1. **Sidebar is always visible** (collapsible to icons only)
2. **No right-side CopilotSidebar** - chat is the main content
3. **Artifacts render inline** in chat stream (like Claude)
4. **Detail panels slide in** from right when needed (3D viewer, etc)
5. **Dark theme by default** matching Antigravity aesthetic
6. **@ mentions** for referencing tools/artifacts in input
7. **Keyboard shortcuts** for power users

---

## Migration Strategy

1. Create new Sidebar component alongside existing code
2. Add feature flag to switch between old/new layout
3. Migrate one section at a time (Workspaces → Information → Evolution)
4. Test with real workflows
5. Remove old code once stable

---

## Open Questions

1. Should Computation area split when viewing 3D structures?
   - Option A: Full-width 3D viewer replaces chat
   - Option B: Split view (chat left, 3D right)
   - Option C: 3D opens in modal/overlay

2. How to handle multiple conversations?
   - Tab-based within workspace?
   - List in sidebar with quick preview?

3. Model selector - which models to expose?
   - Gemini 2.0 Flash (fast)
   - Gemini 2.0 Pro (better)
   - Custom/fine-tuned?

---

## Success Metrics

- [ ] Sidebar renders with all sections
- [ ] Workspaces show agent/project list
- [ ] Information panel shows available tools
- [ ] Evolution panel shows artifacts
- [ ] Chat renders inline artifact cards
- [ ] Input has mode/model selectors
- [ ] Dark theme matches Antigravity aesthetic
- [ ] AG-UI events properly update UI state
