import type { PersonAnalysis } from '../types';

const POSTURE_TEXT: Record<string, string> = {
  standing: 'Standing',
  sitting_or_crouching: 'Sitting or crouching',
  lying: 'Lying or horizontal',
  unknown: 'Not determinable',
};

const ACTIVITY_TEXT: Record<string, string> = {
  low: 'Low',
  moderate: 'Moderate',
  high: 'High',
  unknown: 'Measuring',
};

const HEAD_TEXT: Record<string, string> = {
  left: 'Turned left',
  right: 'Turned right',
  forward: 'Facing camera',
  unknown: 'Not determinable',
};

function titleCase(label: string): string {
  return label.charAt(0).toUpperCase() + label.slice(1);
}

export function PersonCard({ person }: { person: PersonAnalysis }) {
  const { expression, behavior, face, pose } = person;
  const ranked = Object.entries(expression.probabilities).sort((a, b) => b[1] - a[1]);

  return (
    <article className="person-card">
      <header>
        <h3>Person #{String(person.track_id).padStart(2, '0')}</h3>
        <span className="person-confidence">
          {(person.confidence * 100).toFixed(0)}% detection
        </span>
      </header>

      <div className="person-row">
        <div className="person-block">
          <p className="block-title">Facial expression</p>
          {expression.available && ranked.length > 0 ? (
            <ul className="expression-bars">
              {ranked.slice(0, 4).map(([label, probability]) => (
                <li key={label}>
                  <span className="expression-label">{titleCase(label)}</span>
                  <span className="bar">
                    <span className="bar-fill" style={{ width: `${Math.round(probability * 100)}%` }} />
                  </span>
                  <span className="expression-value">{Math.round(probability * 100)}%</span>
                </li>
              ))}
            </ul>
          ) : (
            <p className="block-empty">
              {face.detected
                ? 'Face found, expression model unavailable.'
                : 'No face visible to this camera.'}
            </p>
          )}
        </div>

        <div className="person-block">
          <p className="block-title">Body and movement</p>
          <dl className="readout">
            <div>
              <dt>Posture</dt>
              <dd>{POSTURE_TEXT[behavior.posture] ?? behavior.posture}</dd>
            </div>
            <div>
              <dt>Activity level</dt>
              <dd className={`activity activity-${behavior.activity_level}`}>
                {ACTIVITY_TEXT[behavior.activity_level] ?? behavior.activity_level}
              </dd>
            </div>
            <div>
              <dt>Movement speed</dt>
              <dd>{behavior.movement_speed.toFixed(3)} w/s</dd>
            </div>
            <div>
              <dt>Head direction</dt>
              <dd>{behavior.head_orientation ? HEAD_TEXT[behavior.head_orientation] ?? behavior.head_orientation : 'Not determinable'}</dd>
            </div>
            <div>
              <dt>Keypoints</dt>
              <dd>{pose.visible_keypoints} of 17 visible</dd>
            </div>
            <div>
              <dt>Face detection</dt>
              <dd>
                {face.detected
                  ? `Found (${face.source}, ${(face.confidence * 100).toFixed(0)}%)`
                  : 'Not found'}
              </dd>
            </div>
            <div>
              <dt>In view for</dt>
              <dd>{behavior.time_visible_seconds.toFixed(1)} s</dd>
            </div>
          </dl>
        </div>
      </div>
    </article>
  );
}

export function PersonList({ people }: { people: PersonAnalysis[] }) {
  return (
    <section className="panel person-panel">
      <h2>Tracked people</h2>
      {people.length === 0 ? (
        <p className="panel-empty">
          Nobody in view. People appear here with a tracking ID as soon as the
          detector finds them.
        </p>
      ) : (
        <div className="person-list">
          {people
            .slice()
            .sort((a, b) => a.track_id - b.track_id)
            .map((person) => (
              <PersonCard key={person.track_id} person={person} />
            ))}
        </div>
      )}
    </section>
  );
}
