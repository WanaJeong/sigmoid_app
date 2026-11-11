import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="Sigmoid Activation Function", layout="wide")
st.title("Sigmoid Activation Function")
st.write("Sigmoid converts an input value into an output between 0 and 1. It is commonly used for binary classification.")
st.latex(r"f(x) = \\frac{1}{1 + e^{-x}}")

st.sidebar.header("Input Controls")
x = st.sidebar.slider("Choose an input value (x)", -10.0, 10.0, 2.0, 0.1)
y = 1.0 / (1.0 + np.exp(-x))

c1, c2 = st.columns(2)
c1.metric("Input (x)", f"{x:.2f}")
c2.metric("Sigmoid Output", f"{y:.4f}")

st.subheader("Sigmoid Function Graph")
xgrid = np.linspace(-10, 10, 400)
ygrid = 1.0 / (1.0 + np.exp(-xgrid))
fig, ax = plt.subplots(figsize=(8, 4.5))
ax.plot(xgrid, ygrid, label="Sigmoid", linewidth=2)
ax.scatter([x], [y], s=80, label="Selected input/output")
ax.axhline(0, linewidth=0.8)
ax.axhline(1, linewidth=0.8, linestyle="--")
ax.axvline(0, linewidth=0.8)
ax.set(xlabel="Input (x)", ylabel="Output f(x)", title="Sigmoid Activation Function")
ax.set_ylim(-0.05, 1.05)
ax.grid(True)
ax.legend()
st.pyplot(fig)
plt.close(fig)

st.subheader("Interpretation")
st.write(f"For an input of {x:.2f}, Sigmoid produces an output of {y:.4f}. The output stays between 0 and 1.")
st.info("The Sigmoid function has an S-shaped curve and maps real-valued inputs to the range (0, 1).")
