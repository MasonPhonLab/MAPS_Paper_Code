using DelimitedFiles

function dtw_align(seq, probs, rev=true)

    # offset sequence by 1 since Python indexes at 0
    seq .+= 1
    
    if rev
        seq = reverse(seq)
        probs = probs[reverse(1:size(probs, 1)), :]
    end
    
    function distance(r, c)
        category = seq[r]
        timestep = c
        return probs[timestep, category]
    end
    
    nrow = length(seq)
    ncol = size(probs, 1) # time steps
    
    M = dtw(collect(1:nrow), collect(1:ncol), distance)
    if rev
        M = M[reverse(2:end), reverse(2:end)]
        alignment = rbacktrack(M, reverse(seq))
    else
        alignment = backtrack(M[2:end, 2:end], seq)
    end
    
    # offset sequence by 1 to return to Python style indexing
    alignment .-= 1
    
    if rev return alignment, M end
    return alignment, M[2:end, 2:end]
end

function dtw(S, T, d)
	nrow = length(S)
    ncol = length(T)
    
    M = zeros(nrow+1, ncol+1) .+ Inf64
    durTracker = zeros(nrow+1, ncol+1)
    M[1, 1] = 0
    
    for r in 2:(nrow+1)
        for c in 2:(ncol+1)
            cost = d(r-1, c-1)           
            M[r, c] = cost + min(M[r, c-1], M[r-1, c-1])
        end
    end
    
    return M
end

function dtw(S, T, d, seq)
    nrow = length(S)
    ncol = length(T)
    
    M = zeros(nrow+1, ncol+1) .+ Inf64
    durTracker = zeros(nrow+1, ncol+1)
    M[1, 1] = 0
    
    for r in 2:(nrow+1)
        for c in 2:(ncol+1)
            cost = d(r-1, c-1)
            
            prevCost = min(M[r, c-1], M[r-1, c-1])
            
            if prevCost == M[r, c-1]
                durTracker[r, c] = durTracker[r, c-1] + 1
                prevCost -= pmf(DM, seq[r], durTracker[r, c-1])
                cost += pmf(DM, seq[r], durTracker[r, c])
            else
                durTracker[r, c] = 1
                cost += pmf(DM, seq[r], 1)
            end
            M[r, c] = cost + prevCost
        end
    end
    
    return M
end

function backtrack(M, sequence)
    nrow, ncol = size(M)
    seq = [nrow]
    
    r = nrow
    c = ncol
    
    curr_prob = M[r, c]
    
    while c > 0
        if r == 1
            append!(seq, repeat([1], c-1))
            break
        end
        
        if M[r-1, c-1] > M[r, c-1]
            push!(seq, r)
            c -= 1
        else
            c -= 1
            r -= 1
            
            push!(seq, r)
        end
    end
    
    rs = reverse(seq)
    rs = [sequence[r] for r in rs]
    return rs
end

function rbacktrack(M, sequence)
    writedlm("revM.txt", M, '\t')
    seq = [first(sequence)]
    r = 1
    for c in 2:size(M, 2)
        r += r < size(M, 1) && M[r, c] > M[r+1, c]
        push!(seq, sequence[r])
    end
    return seq
end
