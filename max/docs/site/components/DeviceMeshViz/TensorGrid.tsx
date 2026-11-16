import React, { useRef } from 'react';
import clsx from 'clsx';
import { locateCell, type Cell } from './steps';
import styles from './styles.module.scss';

type TensorGridProps = {
  rows: number;
  columns: number;
  rowOffset?: number;
  activeRowStart?: number;
  activeRowCount?: number;
  device?: number;
  shardRows?: number[];
  colored?: boolean;
  showCoordinates?: boolean;
  localCoordinates?: boolean;
  selected?: Cell;
  interactive?: boolean;
  onSelect?: (cell: Cell) => void;
  label: string;
  describedBy?: string;
};

const ARROW_OFFSETS: Record<string, [number, number]> = {
  ArrowUp: [-1, 0],
  ArrowDown: [1, 0],
  ArrowLeft: [0, -1],
  ArrowRight: [0, 1],
};

export function TensorGrid({
  rows,
  columns,
  rowOffset = 0,
  activeRowStart = rowOffset,
  activeRowCount = rows,
  device,
  shardRows,
  colored = false,
  showCoordinates = false,
  localCoordinates = false,
  selected,
  interactive = false,
  onSelect,
  label,
  describedBy,
}: TensorGridProps) {
  const cellRefs = useRef<Array<HTMLButtonElement | null>>([]);

  function handleKeyDown(
    event: React.KeyboardEvent<HTMLButtonElement>,
    index: number
  ) {
    const offset = ARROW_OFFSETS[event.key];
    if (!offset) return;

    event.preventDefault();
    const row = Math.floor(index / columns);
    const column = index % columns;
    const nextRow = Math.max(0, Math.min(rows - 1, row + offset[0]));
    const nextColumn = Math.max(0, Math.min(columns - 1, column + offset[1]));
    const nextIndex = nextRow * columns + nextColumn;

    onSelect?.({ row: nextRow + rowOffset, column: nextColumn });
    cellRefs.current[nextIndex]?.focus();
  }

  return (
    <div
      className={styles.tensorGrid}
      style={{ gridTemplateColumns: `repeat(${columns}, minmax(0, 1fr))` }}
      role="group"
      aria-label={label}
      aria-describedby={describedBy}
    >
      {Array.from({ length: rows * columns }, (_, index) => {
        const localRow = Math.floor(index / columns);
        const column = index % columns;
        const row = localRow + rowOffset;
        const cell = { row, column };
        const location = locateCell(cell, shardRows);
        const cellDevice = device ?? location.device;
        const isSelected = selected?.row === row && selected.column === column;
        const isActive =
          row >= activeRowStart && row < activeRowStart + activeRowCount;
        let displayValue = '';
        if (isActive) {
          if (showCoordinates) {
            displayValue = localCoordinates
              ? `(${localRow},${column})`
              : `(${row},${column})`;
          } else {
            displayValue = '1.0';
          }
        }
        const accessibleValue = showCoordinates
          ? `x[${row}, ${column}]`
          : `x[${row}, ${column}] = 1.0`;
        const className = clsx(
          styles.tensorCell,
          colored && styles[`device${cellDevice}`],
          showCoordinates && styles.tensorCellCoordinate,
          !isActive && styles.tensorCellMuted,
          isSelected && styles.tensorCellSelected
        );

        if (!interactive) {
          return (
            <span
              className={className}
              key={`${row}-${column}`}
              aria-hidden="true"
            >
              {displayValue}
            </span>
          );
        }

        return (
          <button
            type="button"
            className={className}
            key={`${row}-${column}`}
            ref={(element) => {
              cellRefs.current[index] = element;
            }}
            tabIndex={isSelected ? 0 : -1}
            onMouseEnter={() => onSelect?.(cell)}
            onFocus={() => onSelect?.(cell)}
            onClick={() => onSelect?.(cell)}
            onKeyDown={(event) => handleKeyDown(event, index)}
            aria-pressed={isSelected}
            aria-label={`${accessibleValue}. Maps to x.local_shards[${location.device}][${location.localRow}, ${location.localColumn}].`}
          >
            {displayValue}
          </button>
        );
      })}
    </div>
  );
}
