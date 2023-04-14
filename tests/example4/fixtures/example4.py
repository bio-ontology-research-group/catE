import pandas as pd

a = "http://mowl/A"
b = "http://mowl/B"
c = "http://mowl/C"
d = "http://mowl/D"
e = "http://mowl/E"
f = "http://mowl/F"

bot = "owl:Nothing"

not_a = f"not {a}"
not_b = f"not {b}"
not_c = f"not {c}"
not_d = f"not {d}"

a_and_c = f"{a} and {c}"
a_and_d = f"{a} and {d}"
b_and_c = f"{b} and {c}"
b_and_d = f"{b} and {d}"

a_or_b = f"{a} or {b}"
c_or_d = f"{c} or {d}"
e_or_f = f"{e} or {f}"

a_and_c_and_e = f"{a_and_c} and {e}"
b_and_c_and_e = f"{b_and_c} and {e}"
a_and_d_and_e = f"{a_and_d} and {e}"
b_and_d_and_e = f"{b_and_d} and {e}"
a_and_c_and_f = f"{a_and_c} and {f}"
b_and_c_and_f = f"{b_and_c} and {f}"
a_and_d_and_f = f"{a_and_d} and {f}"
b_and_d_and_f = f"{b_and_d} and {f}"

c1 = f"{a_and_c_and_e} or {b_and_c_and_e} or {a_and_d_and_e} or {b_and_d_and_e} or {a_and_c_and_f} or {b_and_c_and_f} or {a_and_d_and_f} or {b_and_d_and_f}"


c0 = f"{a_or_b} and {c_or_d} and {e_or_f}"

edges = [
    (a_and_c, bot),
    (a_and_d, bot),
    (b_and_c, bot),
    (b_and_d, bot),

    (c, not_a),
    (d, not_a),
    (c, not_b),
    (d, not_b),

    (a, a_or_b),
    (b, a_or_b),
    (d, c_or_d),
    (c, c_or_d),
    (f, e_or_f),
    (e, e_or_f),
 

    (c0, a_or_b),
    (c0, c_or_d),
    (c0, e_or_f),
                                    
    (a_and_c_and_e, bot),
    (b_and_c_and_e, bot),
    (a_and_d_and_e, bot),
    (b_and_d_and_e, bot),
    (a_and_c_and_f, bot),
    (b_and_c_and_f, bot),
    (a_and_d_and_f, bot),
    (b_and_d_and_f, bot),
    (c0, bot),
]

out_file = "ground_truth.csv"
pd.DataFrame(edges).to_csv(out_file, index=False, header=False)
