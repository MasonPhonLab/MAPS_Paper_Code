using CSV, DataFrames, Plots
d = CSV.read("test_errs.csv", DataFrame)
plot(d.boundary, [d.interp, d.nointerp, d.mfa], label=["Interp" "No interp" "MFA"],
  xlabel="Boundary error (ms)", ylabel="Proportion within tolerance",
  lw=1.5, xticks=0:10:100)
savefig("crisp_res.pdf")