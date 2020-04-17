import re

def symbol(symbol):
    for match in re.finditer(r"\[([^]]+)\]", symbol):
        symbol = symbol.replace(match.group(), globals()[match.group()[1:-1]])
    return f'${symbol}$'

def equation(maths): 
    for match in re.finditer(r"\[([^]]+)\]", maths):
        maths = maths.replace(match.group(), globals()[match.group()[1:-1]])
    return f'$${maths}$$'

V_feat = '\overline{f}_i'

V_theta = '\overline{\theta}_y'

V_hist = 'y_{ijs}'

V_v = '\overline{v}_y'

V_u = '\overline{u}_{ys}'

V_l1 = '\lambda_{1y}'

V_l2 = '\lambda_{2ys}'

S_row = "\sum_\limits{y=0}^Y"

S_col = "\sum_\limits{x=0}^X"

S_patch = f"{S_row}{S_col}"

S_adj = "\sum_\limits{j}^4" 

S_scale = "\sum_\limits{s=1}^3"

E_pred = f'{V_feat}{V_theta}'

E_abs = f'| d_i - {E_pred} |'

E_rel = '| d_i(s) - d_j(s) |'

E_l1 = f'{V_v}{V_feat}'

E_l2 = f'{V_u}|{V_hist}|'

E_e1 = f'\dfrac{{{E_abs}}}{{2{V_l1}}}'

E_e2 = f'\dfrac{{{E_rel}}}{{2{V_l2}}}'

E_lap = f'\dfrac{{1}}{{Z}}e^\left(-{S_patch}{E_e1} - {S_scale}{S_patch}{S_adj}{E_e2} \right)'

E_log = f'-{E_e1} - {S_scale}{S_adj}{E_e2}'