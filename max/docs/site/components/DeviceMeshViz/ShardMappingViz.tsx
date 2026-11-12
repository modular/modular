import React, { useId, useState } from 'react';
import { TensorGrid } from './TensorGrid';
import { cellMappingCode, locateCell, type Cell } from './steps';
import styles from './styles.module.scss';

type ShardMappingVizProps = {
  rows: number;
  shardRows: number[];
  initialCell: Cell;
  placementLabel: string;
};

export function ShardMappingViz({
  rows,
  shardRows,
  initialCell,
  placementLabel,
}: ShardMappingVizProps) {
  const captionId = useId();
  const instructionsId = useId();
  const keyboardInstructionsId = useId();
  const [selected, setSelected] = useState<Cell>(initialCell);

  return (
    <figure className={styles.root} aria-describedby={captionId}>
      <p className={styles.instructions} id={instructionsId}>
        Colors show which local shard holds each row. Select a cell for its
        coordinate mapping.
      </p>
      <span className={styles.srOnly} id={keyboardInstructionsId}>
        Use the arrow keys to move between cells. Each cell includes its global
        coordinate and corresponding local shard.
      </span>
      <div className={styles.scene}>
        <div className={styles.globalPanel}>
          <div className={styles.panelHeading}>
            <code>x</code>
            <span className={styles.shape}>({rows}, 4)</span>
          </div>
          <TensorGrid
            rows={rows}
            columns={4}
            shardRows={shardRows}
            colored
            showCoordinates
            selected={selected}
            interactive
            onSelect={setSelected}
            label="Cells of x"
            describedBy={`${instructionsId} ${keyboardInstructionsId}`}
          />
        </div>

        <div className={styles.connector} aria-hidden="true">
          <span className={styles.connectorLabel}>{placementLabel}</span>
          <span className={styles.connectorLine} />
          <span className={styles.connectorArrow}>→</span>
        </div>

        <div className={styles.devices}>
          {shardRows.map((localRows, device) => {
            const rowOffset = shardRows
              .slice(0, device)
              .reduce((total, value) => total + value, 0);
            const location = locateCell(selected, shardRows);

            return (
              <div
                className={`${styles.device} ${styles[`device${device}`]}`}
                key={device}
              >
                <div className={styles.panelHeading}>
                  <code>x.local_shards[{device}]</code>
                  <span className={styles.shape}>({localRows}, 4)</span>
                </div>
                <TensorGrid
                  rows={localRows}
                  columns={4}
                  rowOffset={rowOffset}
                  device={device}
                  shardRows={shardRows}
                  colored
                  showCoordinates
                  localCoordinates
                  selected={selected}
                  label={`x.local_shards[${device}]`}
                />
                <span className={styles.srOnly}>
                  {location.device === device
                    ? `The selected global cell maps to local row ${location.localRow}.`
                    : ''}
                </span>
              </div>
            );
          })}
        </div>
      </div>

      <figcaption className={styles.mapping} id={captionId}>
        <code aria-live="polite" aria-atomic="true">
          {cellMappingCode(selected, shardRows)}
        </code>
      </figcaption>
    </figure>
  );
}
