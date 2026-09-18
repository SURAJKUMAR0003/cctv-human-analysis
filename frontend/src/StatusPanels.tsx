import type { ConnectionState, FrameAnalysis, ModelInfo, PipelineStage } from '../types';

interface MetricProps {
  analysis: FrameAnalysis | null;
  connection: ConnectionState;
  cameraConnected: boolean;
}

const CONNECTION_TEXT: Record<ConnectionState, string> = {
  connecting: 'Connecting',
  live: 'Live',
  reconnecting: 'Reconnecting',
  offline: 'No data',
};

export function MetricStrip({ analysis, connection, cameraConnected }: MetricProps) {
  const metrics = [
    {
      label: 'People detected',
      value: analysis ? String(analysis.people_count) : '--',
      sub: analysis ? `${analysis.people.length} tracked` : 'waiting for frames',
    },
    {
      label: 'Analysis rate',
      value: analysis ? analysis.fps.toFixed(1) : '--',
      sub: analysis ? `capture ${analysis.capture_fps.toFixed(1)} fps` : 'fps',
    },
    {
      label: 'Frame time',
      value: analysis ? `${Math.round(analysis.processing_ms)}` : '--',
      sub: 'ms per analysed frame',
    },
    {
      label: 'Camera',
      value: cameraConnected ? 'Connected' : 'Disconnected',
      sub: analysis?.source ?? '',
      tone: cameraConnected ? 'ok' : 'alert',
    },
    {
      label: 'Server feed',
      value: CONNECTION_TEXT[connection],
      sub: 'websocket /ws/live',
      tone: connection === 'live' ? 'ok' : connection === 'offline' ? 'alert' : 'warn',
    },
  ];

  return (
    <div className="metric-strip">
      {metrics.map((metric) => (
        <div key={metric.label} className={`metric tone-${metric.tone ?? 'plain'}`}>
          <p className="metric-value">{metric.value}</p>
          <p className="metric-label">{metric.label}</p>
          {metric.sub && <p className="metric-sub">{metric.sub}</p>}
        </div>
      ))}
    </div>
  );
}

const STAGE_LABEL: Record<string, string> = {
  pose: 'Detection and pose',
  tracking: 'Person tracking',
  face: 'Face detection',
  expression: 'Facial expression',
  behavior: 'Movement features',
};

export function PipelinePanel({ stages }: { stages: PipelineStage[] }) {
  return (
    <section className="panel">
      <h2>AI pipeline</h2>
      {stages.length === 0 ? (
        <p className="panel-empty">Pipeline status arrives with the first frame.</p>
      ) : (
        <ul className="stage-list">
          {stages.map((stage) => (
            <li key={stage.name}>
              <span className={`dot dot-${stage.status}`} aria-hidden />
              <span className="stage-name">{STAGE_LABEL[stage.name] ?? stage.name}</span>
              <span className="stage-detail" title={stage.detail}>
                {stage.detail}
              </span>
              <span className="stage-time">
                {stage.last_duration_ms > 0 ? `${stage.last_duration_ms.toFixed(0)} ms` : ''}
              </span>
            </li>
          ))}
        </ul>
      )}
    </section>
  );
}

export function ModelPanel({ models }: { models: ModelInfo[] }) {
  if (models.length === 0) return null;
  return (
    <section className="panel">
      <h2>Models</h2>
      <ul className="model-list">
        {models.map((model) => (
          <li key={model.key}>
            <span className={`dot ${model.loaded ? 'dot-ok' : model.present ? 'dot-degraded' : 'dot-error'}`} aria-hidden />
            <span className="model-name">{model.name}</span>
            <span className="model-state">
              {model.loaded ? 'loaded' : model.present ? 'on disk, not loaded' : 'missing'}
            </span>
          </li>
        ))}
      </ul>
      {models.some((model) => !model.present) && (
        <p className="panel-hint">
          Missing files stay missing until you run{' '}
          <code>python scripts/download_models.py</code>. Stages without weights are
          switched off rather than estimated.
        </p>
      )}
    </section>
  );
}
