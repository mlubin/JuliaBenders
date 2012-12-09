#load("smpsreader")
load("/home/mlubin/hdd/jlBenders/smpsreader.jl")
load("/home/mlubin/hdd/jlBenders/solveextensive.jl")


function solveBenders(d::SMPSData, nscen::Integer)

    scenarioData = monteCarloSample(d,1:nscen)

    stage1sol = solveExtensive(d,1)
    
    clpmaster = ClpModel()

    clpsubproblem = ClpModel()
    clp_set_log_level(clpsubproblem,0)

    ncol1 = d.firstStageData.ncol
    nrow1 = d.firstStageData.nrow
    nrow2 = d.secondStageTemplate.nrow
    # add \theta variables for cuts
    thetaidx = [(ncol1+1):(ncol1+nscen)]
    clp_load_problem(clpmaster, d.Amat, d.firstStageData.collb,
        d.firstStageData.colub, d.firstStageData.obj, d.firstStageData.rowlb,
        d.firstStageData.rowub)
    zeromat = SparseMatrixCSC(int32(nrow1),int32(nscen),ones(Int32,nscen+1),Int32[],Float64[])
    clp_add_columns(clpmaster, -1e8*ones(nscen), Inf*ones(nscen),
        (1/nscen)*ones(nscen), zeromat)

    clp_load_problem(clpsubproblem, d.Wmat, d.secondStageTemplate.collb,
        d.secondStageTemplate.colub, d.secondStageTemplate.obj,
        d.secondStageTemplate.rowlb, d.secondStageTemplate.rowub)

    thetasol = -1e8*ones(nscen)

    converged = false
    while true
        Tx = d.Tmat*stage1sol
        # solve benders subproblems
        nviolated = 0
        #print("current solution is [")
        #for i in 1:ncol1
        #    print("$(stage1sol[i]),")
        #end
        #println("]")
        for s in 1:nscen
            clp_chg_row_lower(clpsubproblem, scenarioData[s][1]-Tx)
            clp_chg_row_upper(clpsubproblem, scenarioData[s][2]-Tx)
            clp_initial_solve(clpsubproblem)
            # don't handle infeasible subproblems yet
            @assert clp_is_proven_optimal(clpsubproblem)
            optval = clp_get_obj_value(clpsubproblem)
            duals = clp_dual_row_solution(clpsubproblem)
            subgrad = zeros(ncol1)
            for i in 1:nrow2
                status = clp_get_row_status(clpsubproblem,i)
                if (status == 1) # basic
                    continue
                end
                for k in 1:ncol1
                    subgrad[k] += -duals[i]*d.Tmat[i,k]
                end
            end
            #println("For scen $s, optval is $optval and model value is $(thetasol[s])")
            if (optval > thetasol[s] + 1e-7)
                nviolated += 1
                #print("adding cut: [")
                # add (0-based) cut to master
                cutvec = Float64[]
                cutcolidx = Int32[]
                for k in 1:ncol1
                 #   print("$(subgrad[k]),")
                    if abs(subgrad[k]) > 1e-10
                        push(cutvec,-subgrad[k])
                        push(cutcolidx,k-1)
                    end
                end
                #println("]")
                push(cutvec,1.)
                push(cutcolidx,ncol1+s-1)
                cutnnz = length(cutvec)
                cutlb = optval-dot(subgrad,stage1sol)

                clp_add_rows(clpmaster, 1, [cutlb], [1e25], Int32[0,cutnnz], cutcolidx, cutvec)
            end

        end

        if nviolated == 0
            break
        end
        println("Generated $nviolated violated cuts")
        # resolve master
        clp_initial_solve(clpmaster)
        @assert clp_is_proven_optimal(clpmaster)
        sol = clp_get_col_solution(clpmaster)
        stage1sol = sol[1:ncol1]
        thetasol = sol[(ncol1+1):end]
    end

    println("Optimal objective is: $(clp_get_obj_value(clpmaster))")

end

# how do you do equivalent of __name__ == "__main__"?
if true
    s = ARGS[1]
    nscen = int(ARGS[2])
    d = SMPSData(strcat(s,".cor"),strcat(s,".tim"),strcat(s,".sto"))
    solveBenders(d,nscen)
end
