import { ShardMappingViz } from './ShardMappingViz';

export function LocalShardsViz() {
  return (
    <ShardMappingViz
      rows={8}
      shardRows={[4, 4]}
      initialCell={{ row: 5, column: 2 }}
      placementLabel="Sharded(0)"
    />
  );
}

export function UnevenShardsViz() {
  return (
    <ShardMappingViz
      rows={10}
      shardRows={[3, 3, 2, 2]}
      initialCell={{ row: 8, column: 2 }}
      placementLabel="Sharded(0), Sharded(0)"
    />
  );
}

export { ReshardingViz } from './ReshardingViz';
