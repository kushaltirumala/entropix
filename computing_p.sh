
for seed_var in 44 64 104 84 94 93 203 303 53 113
do
    echo "on variable $seed_var"
	python3 computing_p.py --seed=$seed_var
done

# python3 computing_p.py --seed=40