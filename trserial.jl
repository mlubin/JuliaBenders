require("bendersserial")

# l_infty trust region
function setTR(c::ClpModel, center, nominallb, nominalub, radius)
    newlb = copy(lb)
    newub = copy(ub)
    for i in 1:length(center)
        newlb[i] = max(nominallb[i],-radius+center[i])
        newub[i] = min(nomunalub[i],radius+center[i])
    end
    clp_chg_column_lower(c,newlb)
    clp_chg_column_upper(c,newub)
end


function solveTRSerial(d::SMPSData, nscen::Integer)

    scenarioData = monteCarloSample(d,1:nscen)

    stage1sol = solveExtensive(d,1)
    
    clpmaster = ClpModel()
    clp_set_log_level(clpmaster,0)
    setGlobalProbData(d)
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

    #thetasol = -1e10*ones(nscen)
    const tr_max = 1000.
    const xi = 1e-4 # parameter related to accepting new iterate
    tr_radius = max(1,0.2*norm(stage1sol,Inf))

    # initialize cuts for model
    begin
        Tx = d.Tmat*stage1sol
        majorobjective = dot(stage1sol,d.firstStageData.obj)
        for s in 1:nscen
            optval, subgrad = solveSubproblem(scenarioData[s][1]-Tx,scenarioData[s][2]-Tx)
            majorobjective += optval/nscen
            addCut(clpmaster, optval, subgrad, stage1sol, s)
        end
    end


    converged = false
    nmajoriter = 0
    nsolves = 0
    mastertime = 0.
    tr_counter = 0
    while true

        # resolve master
        t = time()
        clp_initial_solve(clpmaster)
        mastertime += time() - t
        @assert clp_is_proven_optimal(clpmaster)
        sol = clp_get_col_solution(clpmaster)
        minorstage1sol = sol[1:ncol1]
        #minorthetasol = sol[(ncol1+1):end]
        modelobjective = clp_get_obj_value(clpmaster)
        minorobjective = dot(minorstage1sol,d.firstStageData.obj)
        
        nsolves += 1

        # convergence test
        println("major objective: $majorobjective model obj: $modelobjective")
        if majorobjective - modelobjective < 1e-7(1+abs(majorobjective))
            stepnorm = norm(stage1sol-minorstage1sol,Inf) 
            if abs(stepnorm-tr_radius) < 1e-7
                println("Passed convergence test but TR active")
                tr_radius = min(tr_max,2tr_radius)
                println("Enlarged trust region radius to $tr_radius")
            else
                println("Converged, stepnorm = $stepnorm, tr_radius = $tr_radius")
                break
            end 
        end


        Tx = d.Tmat*minorstage1sol
        # solve benders subproblems
        #print("current solution is [")
        #for i in 1:ncol1
        #    print("$(stage1sol[i]),")
        #end
        #println("]")
        for s in 1:nscen
            optval, subgrad = solveSubproblem(scenarioData[s][1]-Tx,scenarioData[s][2]-Tx)
            minorobjective += optval/nscen
            #if (optval > thetasol[s] + 1e-7)
                addCut(clpmaster, optval, subgrad, minorstage1sol, s)
            #end
        end
        # accept iterate?
        if minorobjective <= majorobjective - xi*(majorobjective - modelobjective)
            println("Accepted new iterate, finished major iteration $nmajoriter")
            nmajoriter += 1
            majorobjective = minorobjective
            tr_counter = 0
            if abs(norm(stage1sol-minorstage1sol,Inf)-tr_radius) < 1e-7 && minorobjective <= majorobjective - 0.5(majorobjective - modelobjective) 
                # enlarge trust radius
                tr_radius = min(tr_max,2tr_radius)
                println("Enlarged trust region radius to $tr_radius")
            end
            stage1sol = minorstage1sol
        else # didn't accept iterate, maybe reduce trust region radius
            rho = min(1,tr_radius)*(minorobjective-majorobjective)/(majorobjective-modelobjective)
            println("Rejected iterate, rho = $rho")
            if rho > 0
                tr_counter += 1
            elseif rho > 3 or (tr_counter >= 3 && 1 < rho <= 3)
                tr_radius = (1/min(rho,4))*tr_radius
                tr_counter = 0
                prinln("Shrunk trust region radius to $tr_radius")
            end
        end


        
        
    end

    println("Optimal objective is: $(clp_get_obj_value(clpmaster)), $nmajoriter major iterations, master solved $nsolves times")
    println("Time in master: $mastertime sec")

end

