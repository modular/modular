export type Cell = {
  row: number;
  column: number;
};

export function locateCell(cell: Cell, shardRows: number[] = [4, 4]) {
  let remainingRow = cell.row;

  for (let device = 0; device < shardRows.length; device += 1) {
    if (remainingRow < shardRows[device]) {
      return {
        device,
        localRow: remainingRow,
        localColumn: cell.column,
      };
    }
    remainingRow -= shardRows[device];
  }

  throw new RangeError(`Row ${cell.row} is outside the global tensor.`);
}

export function cellMappingCode(cell: Cell, shardRows?: number[]) {
  const { device, localRow, localColumn } = locateCell(cell, shardRows);
  return `x[${cell.row}, ${cell.column}] → x.local_shards[${device}][${localRow}, ${localColumn}]`;
}
