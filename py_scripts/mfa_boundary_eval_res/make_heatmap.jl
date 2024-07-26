using Plots
using CSV
using DataFrames

function main()

	fname = ARGS[1]
	d = CSV.read(fname, DataFrame)
	labels = sort(collect(Set(d.next_segment_category)))
	cat2num = Dict(x => i for (i, x) in enumerate(labels))
	
	m = Array{Union{Float64,Missing},2}(missing, 7, 7)
	for r in eachrow(d)
		row_I = cat2num[r.segment_category]
		col_I = cat2num[r.next_segment_category]
		m[row_I, col_I] = r.err
	end
	
	heatmap(1:7, 1:7, m, xticks=(1:7, labels), yticks=(1:7, labels), clims=(0, 50),
		xlab="Following segment", ylab="First segment", size=(800, 400),
		left_margin=2Plots.mm, bottom_margin=2Plots.mm, yaxis=:flip, grid=false,
		colorbar_title="\nMedian abs error (ms)", right_margin=5Plots.mm,
		title="MFA")
		
	savefig("mfa_by_category_errors.pdf")
end

main()
