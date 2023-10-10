using CSV, DataFrames, Plots
d = CSV.read("sparse_test_errs.csv", DataFrame)
plot(d.boundary, [d.interp, d.nointerp, d.mfa], label=["Interp" "No interp" "MFA"],
  xlabel="Boundary error (ms)", ylabel="Proportion within tolerance",
  title="Sparse models", lw=1.5, xticks=0:10:100)
savefig("sparse_res.pdf")