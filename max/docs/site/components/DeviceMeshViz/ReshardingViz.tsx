import React, { useId, useRef, useState } from 'react';
import clsx from 'clsx';
import { TensorGrid } from './TensorGrid';
import styles from './styles.module.scss';

const PLACEMENTS = [
  { label: 'Sharded(0)', replicated: false },
  { label: 'Replicated()', replicated: true },
];

export function ReshardingViz() {
  const captionId = useId();
  const placementLabelId = useId();
  const optionRefs = useRef<Array<HTMLButtonElement | null>>([]);
  const [replicated, setReplicated] = useState(false);
  const localRows = replicated ? 8 : 4;
  const tensorName = replicated ? 'y' : 'x';

  function handleKeyDown(event: React.KeyboardEvent<HTMLDivElement>) {
    if (event.key !== 'ArrowLeft' && event.key !== 'ArrowRight') return;

    event.preventDefault();
    const nextReplicated = event.key === 'ArrowRight';
    setReplicated(nextReplicated);
    optionRefs.current[Number(nextReplicated)]?.focus();
  }

  return (
    <figure className={styles.root} aria-describedby={captionId}>
      <div className={styles.placementControl}>
        <span className={styles.placementLabel} id={placementLabelId}>
          Placement
        </span>
        <div
          className={styles.placementToggle}
          role="group"
          aria-labelledby={placementLabelId}
          onKeyDown={handleKeyDown}
        >
          {PLACEMENTS.map((placement, index) => (
            <button
              type="button"
              className={clsx(
                styles.placementOption,
                replicated === placement.replicated &&
                  styles.placementOptionActive
              )}
              key={placement.label}
              ref={(element) => {
                optionRefs.current[index] = element;
              }}
              tabIndex={replicated === placement.replicated ? 0 : -1}
              aria-pressed={replicated === placement.replicated}
              onFocus={() => setReplicated(placement.replicated)}
              onClick={() => setReplicated(placement.replicated)}
            >
              {placement.label}
            </button>
          ))}
        </div>
      </div>

      <div className={styles.reshardScene}>
        <div className={styles.panelHeading}>
          <code>{tensorName}.local_shards</code>
          <span className={styles.shape}>
            {replicated ? 'Replicated()' : 'Sharded(0)'}
          </span>
        </div>
        <div className={styles.reshardDevices}>
          {[0, 1].map((device) => (
            <React.Fragment key={device}>
              {device === 1 && (
                <div
                  className={clsx(
                    styles.gatherCue,
                    !replicated && styles.gatherCueHidden
                  )}
                  aria-hidden="true"
                >
                  <span>allgather()</span>
                  <span className={styles.gatherArrow}>↔</span>
                </div>
              )}
              <div className={styles.device}>
                <div className={styles.panelHeading}>
                  <code>[{device}]</code>
                  <span className={styles.shape}>({localRows}, 4)</span>
                </div>
                <TensorGrid
                  rows={8}
                  columns={4}
                  activeRowStart={replicated ? 0 : device * 4}
                  activeRowCount={localRows}
                  showCoordinates
                  label={`${tensorName}.local_shards[${device}]`}
                />
              </div>
            </React.Fragment>
          ))}
        </div>
      </div>

      <figcaption className={styles.mapping} id={captionId}>
        <span className={styles.srOnly}>
          {replicated
            ? 'transfer_to() returns y with all 8 rows on each device. x remains Sharded(0).'
            : 'Each device holds one 4-row shard of x.'}
        </span>
        <code aria-live="polite" aria-atomic="true">
          {replicated
            ? 'y = transfer_to(x, DeviceMapping(mesh, (Replicated(),)))'
            : 'x: Sharded(0)'}
        </code>
      </figcaption>
    </figure>
  );
}
