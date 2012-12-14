require("clp")



# This is designed as a reader for the small subset of SMPS used by the problems available at
# http://pages.cs.wisc.edu/~swright/stochastic/sampling/. The SSN problem was used as the template.
# It can only handle two-stage problems where each r.v. is an independent discrete r.v.
# Can only deal with randomness in RHS.
# Does not handle the scenario format.

type BlockData
    ncol::Int
    nrow::Int
    collb::Vector{Float64}
    colub::Vector{Float64}
    obj::Vector{Float64}
    rowlb::Vector{Float64}
    rowub::Vector{Float64}
    colname::Vector{String}
    rowname::Vector{String}
end

type SMPSData
    firstStageData::BlockData
    secondStageTemplate::BlockData
    Tmat::SparseMatrixCSC
    Wmat::SparseMatrixCSC
    Amat::SparseMatrixCSC
    randomIdx::Vector{Int} # indices (wrt 2nd stage) of 2nd stage variables that are random
    randomValues::Vector{Vector{Float64}} # discrete values each r.v. can take
    randomProbabilities::Vector{Vector{Float64}} # corresponding probabilities
end

function SMPSData(cor::String,tim::String,sto::String) 
    
    reader = ClpModel()
    clp_read_mps(reader,cor,true,false)
    collb = clp_get_col_lower(reader)
    colub = clp_get_col_upper(reader)
    obj = clp_get_obj_coefficients(reader)
    rowlb = clp_get_row_lower(reader)
    rowub = clp_get_row_upper(reader)
    A = clp_get_constraint_matrix(reader)
    ncol = size(collb,1)
    nrow = size(rowlb,1)

    colname = String[]
    rowname = String[]
    for i in 1:ncol
        push(colname,clp_column_name(reader,i))
    end
    for i in 1:nrow
        push(rowname,clp_row_name(reader,i))
    end


    # parse tim file
    ft = open(tim,"r")
    line = readline(ft)
    @assert search(line,"TIME") != (0,0)
    line = readline(ft)
    @assert search(line,"PERIODS") != (0,0)
    line = readline(ft) # start of 1st-stage block
    sp = split(line)
    c,r = sp[1],sp[2]
    @assert c == colname[1]
    @assert r == rowname[1]
    line = readline(ft) # start of 2nd-stage block
    sp = split(line)
    c,r = sp[1],sp[2]
    colstart2 = find(colname .== c)[1]
    rowstart2 = find(rowname .== r)[1]

    ncol1 = colstart2-1
    ncol2 = ncol - ncol1

    nrow1 = rowstart2-1
    nrow2 = nrow - nrow1

    firstStageData = BlockData(ncol1,nrow1,collb[1:ncol1],colub[1:ncol1],obj[1:ncol1],rowlb[1:nrow1],rowub[1:nrow1],colname[1:ncol1],rowname[1:nrow1])

    secondStageTemplate = BlockData(ncol2,nrow2,collb[ncol1+1:end],colub[ncol1+1:end],obj[ncol1+1:end],rowlb[nrow1+1:end],rowub[nrow1+1:end],colname[ncol1+1:end],rowname[nrow+1:end])

    Amat = A[1:nrow1,1:ncol1]
    Tmat = A[nrow1+1:end,1:ncol1]
    Wmat = A[nrow1+1:end,ncol1+1:end]
    
    fs = open(sto,"r")
    line = readline(fs)
    @assert search(line,"STOCH")[1] == 1
    line = readline(fs)
    @assert search(line,"INDEP")[1] == 1
    @assert search(line,"DISCRETE") != (0,0)

    idxmap = -ones(Int,nrow2)
    randomIdx = Int[]
    randomValues = Vector{Float64}[]
    randomProbabilities = Vector{Float64}[]



    line = readline(fs)
    while search(line,"ENDATA") == (0,0)
        sp = split(line)
        col,row = sp[1],sp[2]
        val,p = float(sp[3]),float(sp[end])
        @assert search(col,"RHS")[1] == 1
        rowidx = find(rowname[rowstart2:end] .== row)[1]
        @assert 0 <= rowidx <= nrow2
        if idxmap[rowidx] == -1 # haven't seen before
            push(randomIdx,rowidx)
            push(randomValues,[val])
            push(randomProbabilities,[p])
            idxmap[rowidx] = size(randomIdx,1)
        else
            idx = idxmap[rowidx]
            push(randomValues[idx],val)
            push(randomProbabilities[idx],p)
        end
        line = readline(fs)
    end

    # verify probabilities
    nscen = 1.
    for i in 1:length(randomIdx)
        p = 0.
        for k in 1:length(randomProbabilities[i])
            p += randomProbabilities[i][k]
        end
        nscen *= length(randomProbabilities[i])
        @assert abs(1.-p) < 1e-7
    end
    println("$nscen total possible scenarios")

    SMPSData(firstStageData,secondStageTemplate,Tmat,Wmat,Amat,randomIdx,randomValues,randomProbabilities)

end

function monteCarloSample(d::SMPSData,scenariosWanted)
    
    srand(10) # for reproducibility

    out = Array((Vector{Float64},Vector{Float64}),0)
    maxscen = max(scenariosWanted)
    for s in 1:maxscen
        if !contains(scenariosWanted,s)
            # advance rng
            x = rand(length(d.randomIdx))
            continue
        end
        
        rowlb = copy(d.secondStageTemplate.rowlb)
        rowub = copy(d.secondStageTemplate.rowub)
        for i in 1:length(d.randomIdx)
            idx = d.randomIdx[i]
            sample = rand()
            p = 0.
            local k
            for k in 1:length(d.randomProbabilities[i])
                if (p + d.randomProbabilities[i][k] >= sample) 
                    break
                else 
                    p += d.randomProbabilities[i][k]
                end
            end
            val = d.randomValues[i][k]
            if (rowub[idx] == rowlb[idx])
                rowlb[idx] = val
                rowub[idx] = val
            elseif (rowub[idx] < 1e25)
                # don't handle ranged rows
                @assert rowlb[idx] < -1e25
                rowub[idx] = val
            else
                @assert rowlb[idx] > -1e25
                @assert rowub[idx] > 1e25
                rowlb[idx] = val
            end
        end
        push(out,(rowlb,rowub))
    end

    return out
end

