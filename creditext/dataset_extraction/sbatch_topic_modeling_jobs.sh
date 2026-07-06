# batch_pairs=(
# 0	10000
# 10000 20000
# 20000 30000
# 30000 40000
# 40000 50000
# 50000 60000
# 60000 70000
# 70000 80000
# 80000 90000
# )

batch_pairs=(
0	1
1   2
2   3
3   4
4   5
5   6
6   7
7   8
8   9
9   10
)

# task=TopicModeling
# # task=PklToParquet
# for ((i=0; i<${#batch_pairs[@]}; i+=2)); do
#     start=$((batch_pairs[i]))
#     end=$((batch_pairs[i+1]-1))
#     sbatch topic_modeling.sh $start $end  $task
# done

# task=buildIndex
task=domainRel
sbatch topic_modeling.sh 0 90000 $task
