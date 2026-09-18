import { useEffect, useRef, useState } from 'react';
import type { ConnectionState, FrameAnalysis } from '../types';

interface Props {
  apiBase: string;
  connection: ConnectionState;
  analysis: FrameAnalysis | null;
  cameraConnected: boolean;
  cameraError: string;
}

/**
 * Shows the annotated MJPEG stream produced by the backend (boxes, track IDs
 * and pose skeletons are drawn server-side so the overlay can never drift out
 * of sync with the frame it describes).
 */
export function VideoPanel({ apiBase, connection, analysis, cameraConnected, cameraError }: Props) {
  const [streamKey, setStreamKey] = useState(0);
  const [imageFailed, setImageFailed] = useState(false);
  const imageRef = useRef<HTMLImageElement | null>(null);

  // Re-subscribe to the stream whenever the camera comes back.
  useEffect(() => {
    if (cameraConnected) {
      setImageFailed(false);
      setStreamKey((key) => key + 1);
    }
  }, [cameraConnected]);

  const streamUrl = `${apiBase}/api/stream.mjpg?v=${streamKey}`;
  const showStream = cameraConnected && !imageFailed;

  return (
    <section className="video-panel" aria-label="Live camera">
      <div className="video-frame">
        {showStream ? (
          <img
            ref={imageRef}
            key={streamKey}
            src={streamUrl}
            alt="Live camera with detection overlay"
            onError={() => setImageFailed(true)}
          />
        ) : (
          <div className="video-placeholder">
            <p className="placeholder-title">
              {connection === 'offline' || connection === 'reconnecting'
                ? 'Waiting for the analysis server'
                : 'No camera feed'}
            </p>
            <p className="placeholder-body">
              {cameraError ||
                'Check VIDEO_SOURCE in your .env, confirm the camera is not in use by another application, then restart the backend.'}
            </p>
            <button type="button" onClick={() => { setImageFailed(false); setStreamKey((k) => k + 1); }}>
              Retry the stream
            </button>
          </div>
        )}

        {analysis && showStream && (
          <div className="video-badge">
            <span className="badge-count">{analysis.people_count}</span>
            <span className="badge-label">
              {analysis.people_count === 1 ? 'person in view' : 'people in view'}
            </span>
          </div>
        )}
      </div>

      <div className="video-meta">
        <span>{analysis?.source ?? 'source unknown'}</span>
        {analysis && analysis.frame_width > 0 && (
          <span>
            {analysis.frame_width} x {analysis.frame_height}
          </span>
        )}
        <span>frame {analysis?.frame_id ?? 0}</span>
      </div>
    </section>
  );
}
