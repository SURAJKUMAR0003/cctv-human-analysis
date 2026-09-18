import { useLiveAnalysis } from './hooks/useLiveAnalysis';
import { PersonList } from './components/PersonList';
import { MetricStrip, ModelPanel, PipelinePanel } from './components/StatusPanels';
import { VideoPanel } from './components/VideoPanel';
import './styles.css';

export default function App() {
  const { analysis, connection, cameraConnected, cameraError, stages, models, hello, apiBase } =
    useLiveAnalysis();

  const connectionText =
    connection === 'live'
      ? 'Receiving data'
      : connection === 'connecting'
        ? 'Connecting'
        : connection === 'reconnecting'
          ? 'Reconnecting'
          : 'No data';

  return (
    <div className="app">
      <header className="app-header">
        <div>
          <h1>Human analysis</h1>
          <p className="app-subtitle">
            Live person detection, pose, tracking and facial-expression readout
          </p>
        </div>
        <div className="header-status">
          <span className={`status-pill status-${connection}`}>{connectionText}</span>
          {hello && <span className="header-device">{hello.device}</span>}
        </div>
      </header>

      <MetricStrip analysis={analysis} connection={connection} cameraConnected={cameraConnected} />

      <main className="app-body">
        <div className="column-main">
          <VideoPanel
            apiBase={apiBase}
            connection={connection}
            analysis={analysis}
            cameraConnected={cameraConnected}
            cameraError={cameraError}
          />
          <PersonList people={analysis?.people ?? []} />
        </div>

        <aside className="column-side">
          <PipelinePanel stages={stages} />
          <ModelPanel models={models} />
          <section className="panel notice">
            <h2>Reading these numbers</h2>
            <p>
              Expression percentages come from a classifier looking at a cropped face.
              They describe how that face looks to the model in this frame, nothing more.
            </p>
            <p>
              Posture, speed and activity are geometry: how the skeleton sits in the frame
              and how far it moved. None of it measures mood, intent or health, and it
              should not be used that way.
            </p>
          </section>
        </aside>
      </main>
    </div>
  );
}
