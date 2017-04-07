 #!/bin/sh

echo "---------------------------------"
echo "Generate input, run/time program"
echo "---------------------------------"

for i in `seq 1 4`; do
    ./generate/gen -o "input/test$i.out" -v "verification/verify$i.out" -i $((i+11))
    ./compute/trilateration -i "input/test$i.out" -o "output/result$i-a.out" -U 1 -V 1 > "time/result$i-a.out" 2>&1
    ./compute/trilateration -i "input/test$i.out" -o "output/result$i-b.out" -U 2 -V 1 > "time/result$i-b.out" 2>&1
    ./compute/trilateration -i "input/test$i.out" -o "output/result$i-c.out" -U 1 -V 2 > "time/result$i-c.out" 2>&1
    ./compute/trilateration -i "input/test$i.out" -o "output/result$i-d.out" -U 2 -V 2 > "time/result$i-d.out" 2>&1
    ./compute/trilateration -i "input/test$i.out" -o "output/result$i-e.out" -U 0.5 -V 1 > "time/result$i-e.out" 2>&1
    ./compute/trilateration -i "input/test$i.out" -o "output/result$i-f.out" -U 1 -V 0.5 > "time/result$i-f.out" 2>&1
    ./compute/trilateration -i "input/test$i.out" -o "output/result$i-g.out" -U 0.5 -V 0.5 > "time/result$i-g.out" 2>&1
done

echo ""
echo "------------------------------------"
echo "Beginning verification of output"
echo "------------------------------------"

for i in `seq 1 4`; do
    diff -s "output/result$i-a.out" "output/result$i-a.out"
    diff -s "output/result$i-a.out" "output/result$i-b.out"
    diff -s "output/result$i-a.out" "output/result$i-c.out"
    diff -s "output/result$i-a.out" "output/result$i-d.out"
    diff -s "output/result$i-a.out" "output/result$i-e.out"
    diff -s "output/result$i-a.out" "output/result$i-f.out"
    diff -s "output/result$i-a.out" "output/result$i-g.out"
done

echo ""
echo "------------------------------------"
echo "Get and print execution times"
echo "------------------------------------"

for i in `seq 1 4`; do
    echo -e "Configuration $i/A -> \c"
    cat "time/result$i-a.out"
    echo -e "Configuration $i/B -> \c"
    cat "time/result$i-b.out"
    echo -e "Configuration $i/C -> \c"
    cat "time/result$i-c.out"
    echo -e "Configuration $i/D -> \c"
    cat "time/result$i-d.out"
    echo -e "Configuration $i/E -> \c"
    cat "time/result$i-e.out"
    echo -e "Configuration $i/F -> \c"
    cat "time/result$i-f.out"
    echo -e "Configuration $i/G -> \c"
    cat "time/result$i-g.out"
done