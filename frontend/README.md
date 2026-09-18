# Dashboard

React + TypeScript + Vite front end for the analysis backend.

```bash
npm install
npm run dev      # http://localhost:5173
npm run build    # production bundle in dist/
npx tsc --noEmit # typecheck
```

`vite.config.ts` proxies `/api`, `/health` and `/ws` to `http://127.0.0.1:8000`,
so in development everything is same-origin and no CORS setup is needed.

When the dashboard is served from a different origin than the API, set
`VITE_API_BASE` (see `.env.example`) and add that origin to `CORS_ORIGINS` in
the backend `.env`.

## What it shows

- Live annotated video (MJPEG). Boxes, track IDs and skeletons are drawn
  server-side, so the overlay can never drift out of sync with its frame.
- People count, analysis fps, capture fps, frame time, camera state and
  WebSocket state.
- Per-stage pipeline status with timings, and per-model load status.
- One card per tracked person: expression probabilities, posture, activity
  level, movement speed, head direction, visible keypoints, face detection
  state and time in view.

Every value comes from `/ws/live`. There are no placeholder numbers: when a
stage has no result, the card says so in words instead of showing a figure.

## Structure

```
src/
├── types.ts                    mirrors backend/app/models/schemas.py
├── hooks/useLiveAnalysis.ts    WebSocket client with reconnect backoff
├── components/
│   ├── VideoPanel.tsx          MJPEG feed + empty/error states
│   ├── StatusPanels.tsx        metric strip, pipeline status, models
│   └── PersonList.tsx          person cards and expression bars
├── App.tsx
└── styles.css                  design tokens + layout
```

Keep `types.ts` in step with the backend schemas when you change the payload.
