v="13.5"
l="1.23"
R="0.36"
D="0.25"
Dr="0.14"
R_d="16.0"
Drc_str="0"

dim="3"
t_max="10.0"
dt="0.001"
every="1000"

out_dir="."

for n in 100 200 400; do
    if [ "${Drc_str}" = "inf" ]; then
        Drc="numpy.inf"
    else
        Drc="${Drc_str}"
    fi

    out_name="n_${n}_v_${v}_l_${l}_R_${R}_D_${D}_Dr_${Dr}_Rd_${R_d}_Drc_${Drc_str}"
    out_path="${out_dir}/${out_name}"

    args="n=${n}, v=${v}, l=${l}, R=${R}, D=${D}, Dr=${Dr}, R_d=${R_d}, dim=${dim}, t_max=${t_max}, dt=${dt}, out='${out_path}', every=${every}, Dr_c=${Drc}"
    cmd="import numpy,mindrop;mindrop.dropsim(${args})"

    echo ${cmd}
    python -c "${cmd}" &
done

