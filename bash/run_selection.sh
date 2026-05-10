echo "CDing to workdir: $WORKDIR"
cd $WORKDIR

datain="$WORKDIR/resources/datasets"
out="$WORKDIR/resources/outsel"

mkdir -p $out

datasets=(aisopos_ntua_2L)
methods=(bio-is)

# if in subdirectory, nav


for d in ${datasets[@]};
do
    echo "-------------------------------------------------------------------------"
    echo "Running dataset $d"
    echo "-------------------------------------------------------------------------"

    for method in ${methods[@]} 
    do
        echo "\tRunning method $method"
        
        python3 run\_generateSplit.py -d $d -m $method --datain $datain --out $out;
    done;
done;