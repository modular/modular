##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##
#!/bin/bash
shopt -s expand_aliases
source $MODULAR_PATH/utils/aliases.sh
export MODULAR_PATH=$MODULAR_PATH
export MODULAR_PYTHON=$MODULAR_PATH/.derived/autovenv/bin/python3

#-	M: [16, 64, 100, 200, 500, 1000, 2048, 4096, 8192]
#  N: 6144
#  K: 4096
#
#- M: [16, 64, 100, 200, 500, 1000, 2048, 4096, 8192]
#  N: 4096
#  K: 4096
#
#- M: [16, 64, 100, 200, 500, 1000, 2048, 4096, 8192]
#  N: 28672
#  K: 4096
#
#- M: [16, 64, 100, 200, 500, 1000, 2048, 4096, 8192]
#  N: 4096
#  K: 14336
#
#- M: [16, 64, 100, 200, 500, 1000, 2048, 4096, 8192]
#  N: 128256
#  K: 4096
#
m_list=(1 8 16 32 64 128 256 512 1024 2048 4096 8192)
n_list=(6144 4096 28672 4096 128256)
k_list=(4096 4096 4096 14336 4096)

m64=(1024 2048 4096 8192)
x64=(64 128 192 256 320 384 448 512 576 640 704 768 832 896 960)

if [ "$1" == "--stride" ]; then
	stride=$2
	for m in ${m64[@]}; do
		for ((x=stride; x<1024; x+=stride)); do
			echo $x
			val=$((x+m))
			m_list=("${m_list[@]}" $val)
		done
	done
fi

echo ">m_list" ${m_list[@]}

output_dir='llama_tuning/'
mkdir -p $output_dir
kbench --clear-cache

for ((i=0;i<5;i++)); do
	for m in ${m_list[@]}; do
		n=${n_list[$i]}
		k=${k_list[$i]}
		echo $m $n $k
		output_path=$output_dir"/m"$m"_n"$n"_k"$k"_output.txt"
		cmd=" tune_matmul.yaml --tune --cached --verbose "
		cmd+="--param \$M:$m --param N:$n --param K:$k "
		cmd+="--filter \$M:$m --filter N:$n --filter K:$k -o $output_path"
		kbench $cmd
		echo $cmd
	done
done
