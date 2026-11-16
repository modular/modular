import { describe, expect, it } from 'vitest';
import { cellMappingCode, locateCell } from './steps';

describe('DeviceMeshViz coordinate mapping', () => {
  it('maps every global cell to the expected local coordinate', () => {
    for (let row = 0; row < 8; row += 1) {
      for (let column = 0; column < 4; column += 1) {
        expect(locateCell({ row, column })).toEqual({
          device: Math.floor(row / 4),
          localRow: row % 4,
          localColumn: column,
        });
      }
    }
  });

  it('maps uneven shards using their row offsets', () => {
    const shardRows = [3, 3, 2, 2];
    const rowOffsets = [0, 3, 6, 8];

    for (let row = 0; row < 10; row += 1) {
      const device = row < 3 ? 0 : row < 6 ? 1 : row < 8 ? 2 : 3;
      expect(locateCell({ row, column: 2 }, shardRows)).toEqual({
        device,
        localRow: row - rowOffsets[device],
        localColumn: 2,
      });
    }
  });

  it('renders the selected coordinate as MAX indexing syntax', () => {
    expect(cellMappingCode({ row: 5, column: 2 })).toBe(
      'x[5, 2] → x.local_shards[1][1, 2]'
    );
    expect(cellMappingCode({ row: 8, column: 2 }, [3, 3, 2, 2])).toBe(
      'x[8, 2] → x.local_shards[3][0, 2]'
    );
  });

  it('rejects rows outside the global tensor', () => {
    expect(() => locateCell({ row: 10, column: 0 }, [3, 3, 2, 2])).toThrow(
      'Row 10 is outside the global tensor.'
    );
  });
});
