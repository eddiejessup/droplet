import numpy as np

s_I = 0.5
v_I = 0.02

M_in_P_L = 25.0
M_in_P_T = 1.3

def I_to_P_l(n=1.0, s=s_I):
	return n * (2.0 / s)

def I_to_P_t(n=1.0, s=s_I, v=v_I):
	return n * (v / (5.0 * s))

def P_to_I_l(n=1.0, s=s_I):
	return n / I_to_P_l(1.0, s)

def P_to_I_t(n=1.0, s=s_I, v=v_I):
	return n / I_to_P_t(1.0, s, v)


def M_to_P_l(n=1.0, M_in_P_l=M_in_P_L):
	return n * M_in_P_l

def M_to_P_t(n=1.0, M_in_P_t=M_in_P_T):
	return n * M_in_P_t

def P_to_M_l(n=1.0, M_in_P_l=M_in_P_L):
	return n / M_to_P_l(1.0, M_in_P_l)

def P_to_M_t(n=1.0, M_in_P_t=M_in_P_T):
	return n / M_to_P_t(1.0, M_in_P_t)


def I_to_M_l(n=1.0, M_in_P_l=M_in_P_L, s=s_I):
	return n * P_to_M_l(I_to_P_l(1.0, s), M_in_P_l)

def I_to_M_t(n=1.0, M_in_P_t=M_in_P_T, s=s_I, v=v_I):
	return n * P_to_M_t(I_to_P_t(1.0, s, v), M_in_P_t)

def M_to_I_l(n=1.0, M_in_P_l=M_in_P_L, s=s_I):
	return n * P_to_I_l(M_to_P_l(1.0, M_in_P_l), s)

def M_to_I_t(n=1.0, M_in_P_t=M_in_P_T, s=s_I, v=v_I):
	return n * P_to_I_t(M_to_P_t(1.0, M_in_P_t), s, v)

def converter(inp, outp, quant, n=1.0):
	if inp == 'M':
		if outp == 'I':
			if quant == 'l':
				return M_to_I_l(n)
			elif quant == 't':
				return M_to_I_t(n)
		elif outp == 'P':
			if quant == 'l':
				return M_to_P_l(n)
			elif quant == 't':
				return M_to_P_t(n)
	elif inp == 'I':
		if outp == 'M':
			if quant == 'l':
				return I_to_M_l(n)
			elif quant == 't':
				return I_to_M_t(n)
		elif outp == 'P':
			if quant == 'l':
				return I_to_P_l(n)
			elif quant == 't':
				return I_to_P_t(n)
	elif inp == 'P':
		if outp == 'M':
			if quant == 'l':
				return P_to_M_l(n)
			elif quant == 't':
				return P_to_M_t(n)
		elif outp == 'I':
			if quant == 'l':
				return P_to_I_l(n)
			elif quant == 't':
				return P_to_I_t(n)	
	print('Invalid arguments, sorry.')
	return

def main():
	inp = raw_input("Input units, 'M' (my units), 'I' (Ian's units) or 'P' (physical units) --> ")
	outp = raw_input("Output units, 'M' (my units), 'I' (Ian's units) or 'P' (physical units) --> ")
	quant = raw_input("Time ('t'), length ('l'), diffusion constant (D)? --> ")
	try:
		n = float(raw_input("How much? Blank for 1.0 --> "))
	except ValueError:
		n = 1.0
	
	if quant in ['l', 't']:
		res = converter(inp, outp, quant, n)
	elif quant == 'D':
		res = n * (converter(inp, outp, 'l') ** 2.0 / converter(inp, outp, 't'))
	elif quant == 'alpha':
		res = n * (converter(inp, outp, 'l') ** 4.0 / converter(inp, outp, 't'))

	print('For quantity %s , %f %s = %f %s' % (quant, n, inp, res, outp))

if __name__ == "__main__":
	main()
