require("smpsreader")

function solveExtensive(d::SMPSData, nscen::Integer)

    scenarioData = monteCarloSample(d,1:nscen)

    # do unpleasant work to form constraint matrix
    # [ A   ]
    # [ T W ]
    Amat = d.Amat
    Tmat = d.Tmat
    Wmat = d.Wmat
    ncol1 = d.firstStageData.ncol
    nrow1 = d.firstStageData.nrow
    ncol2 = d.secondStageTemplate.ncol
    nrow2 = d.secondStageTemplate.nrow
    totalNnz = nnz(Amat) + nscen*nnz(Tmat) + nscen*nnz(Wmat)
    totalVars = ncol1 + nscen*ncol2
    totalCons = nrow1 + nscen*nrow2

    colptr = Array(Int32,totalVars+1)
    rowval = Array(Int32,totalNnz)
    nzval = Array(Float64,totalNnz)

    collb = Array(Float64,totalVars)
    colub = Array(Float64,totalVars)
    obj = Array(Float64,totalVars)
    rowlb = Array(Float64,totalCons)
    rowub = Array(Float64,totalCons)


    nnzcount = 0
    for i in 1:ncol1
        collb[i] = d.firstStageData.collb[i]
        colub[i] = d.firstStageData.colub[i]
        obj[i] = d.firstStageData.obj[i]
        colptr[i] = nnzcount+1
        for k in Amat.colptr[i]:(Amat.colptr[i+1]-1)
            nnzcount += 1
            rowval[nnzcount] = Amat.rowval[k]
            nzval[nnzcount] = Amat.nzval[k]
        end
        for z in 1:nscen
            for k in Tmat.colptr[i]:(Tmat.colptr[i+1]-1)
                nnzcount += 1
                rowval[nnzcount] = Tmat.rowval[k]+nrow1+(z-1)*nrow2
                nzval[nnzcount] = Tmat.nzval[k]
            end
        end
    end
    
    for z in 1:nscen
        for i in 1:ncol2
            thiscol = i+ncol1+(z-1)*ncol2
            @assert thiscol <= totalVars
            collb[thiscol] = d.secondStageTemplate.collb[i]
            colub[thiscol] = d.secondStageTemplate.colub[i]
            obj[thiscol] = (1/nscen)*d.secondStageTemplate.obj[i]
            colptr[thiscol] = nnzcount+1
            for k in Wmat.colptr[i]:(Wmat.colptr[i+1]-1)
                nnzcount += 1
                rowval[nnzcount] = Wmat.rowval[k]+nrow1+(z-1)*nrow2
                nzval[nnzcount] = Wmat.nzval[k]
            end
        end
    end

    colptr[totalVars+1] = nnzcount+1
    @assert nnzcount == totalNnz
    
    Aextensive = SparseMatrixCSC(totalCons,totalVars,colptr,rowval,nzval)

    rowlb[1:nrow1] = d.firstStageData.rowlb
    rowub[1:nrow1] = d.firstStageData.rowub
    for z in 1:nscen
        rowoffset = nrow1 + (z-1)*nrow2
        rowlb[(rowoffset+1):(rowoffset+nrow2)] = scenarioData[z][1]
        rowub[(rowoffset+1):(rowoffset+nrow2)] = scenarioData[z][2]
    end

    c = ClpModel()

    clp_load_problem(c,Aextensive,collb,colub,obj,rowlb,rowub)

    clp_initial_solve(c)
    
    return clp_get_col_solution(c)[1:ncol1]
end

