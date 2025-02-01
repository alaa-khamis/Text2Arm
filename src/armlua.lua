-- Helper functions   ------------------------------------------------------------------------------------------

function createPose(position, quaternion)
    assert(#position == 3, "Position must have exactly 3 elements")
    assert(#quaternion == 4, "Quaternion must have exactly 4 elements")
    local pose = {position[1], position[2], position[3], quaternion[1], quaternion[2], quaternion[3], quaternion[4]}
    return pose
end

function concatenateLists(list1, list2)
    for _, value in ipairs(list2) do
        table.insert(list1, value)
    end
    return list1
end

function eulerToQuaternion(eulerAngles)
    -- eulerAngles should be a table: {alpha, beta, gamma}
    local matrix = sim.buildMatrix({0, 0, 0}, eulerAngles)
    local quaternion = sim.getQuaternionFromMatrix(matrix)
    return quaternion
end

function createPassiveShape(config)
    local bufferedConfig = getCurrConfig()
    setConfig(config)

    local list = sim.getObjectsInTree(params.robotBase, sim.sceneobject_shape)
    local lc = table.clone(list)
    list = {}

    local cylinder = sim.getObject('/UR5/Cylinder') -- Remove the base from the passive shape

    for j = 1, #lc do
        if sim.getBoolProperty(lc[j], 'visible') and lc[j] ~= cylinder and lc[j] ~= params.collisionBox then
            list[#list + 1] = lc[j]
        end
    end

    list = sim.copyPasteObjects(list)
    local passiveVizShape = sim.groupShapes(list, true)
    sim.setBoolProperty(passiveVizShape, 'respondable', false)
    sim.setBoolProperty(passiveVizShape, 'dynamic', false)
    sim.setBoolProperty(passiveVizShape, 'collidable', false)
    sim.setBoolProperty(passiveVizShape, 'measurable', false)
    sim.setBoolProperty(passiveVizShape, 'detectable', false)
    sim.setColorProperty(sim.getIntArrayProperty(passiveVizShape, 'meshes')[1], 'color.diffuse', {1, 0, 0})
    sim.setObjectAlias(passiveVizShape, 'passiveVisualizationShape')
    sim.step()

    setConfig(bufferedConfig)
    return passiveVizShape
end

function visualizePath(pathPts, times, numSamples)
    assert(numSamples > 0, "Number of samples must be greater than 0")

    -- Determine evenly spaced time intervals
    local totalTime = times[#times]
    local sampleTimes = {}
    for i = 0, numSamples - 1 do
        sampleTimes[#sampleTimes + 1] = i * totalTime / (numSamples - 1)
    end

    local shapes = {}

    -- Create passive shapes at sampled points
    for _, t in ipairs(sampleTimes) do
        local sampledConfig = sim.getPathInterpolatedConfig(pathPts, times, t)
        shapes[#shapes + 1] = createPassiveShape(sampledConfig)
    end

    print(string.format("Visualized the path with %d passive shapes", numSamples))

    return shapes

end

-- Arm kinematics functions   ------------------------------------------------------------------------------------------

function moveToPose(pose)
    local p = {
        ik = {tip = params.robotTip, target = params.robotTarget, base = params.robotBase, joints = params.joints},
        targetPose = pose,
        maxVel = params.ikMaxVel,
        maxAccel = params.ikMaxAccel,
        maxJerk = params.ikMaxJerk,
    }
    local result = sim.moveToPose(p)
    if not result then
        print("Failed to move the entire arm to pose:", pose)
    end
    return result
end

function setConfig(c)
    for i = 1, #params.joints do
        sim.setJointPosition(params.joints[i], c[i])
    end
    sim.step()
end

function setTargetConfig(c)
    for i = 1, #params.joints do
        sim.setJointTargetPosition(params.joints[i], c[i])
    end
    sim.step()
end

function getCurrConfig()
    local c = {}
    for i = 1, #params.joints do
        c[i] = sim.getJointPosition(params.joints[i])
    end
    return c
end

function findConfigs(pose)
    local ikEnv = simIK.createEnvironment()
    local ikGroup = simIK.createGroup(ikEnv)
    local ikEl, simToIk, ikToSim = simIK.addElementFromScene(ikEnv, ikGroup, params.robotBase, params.robotTip, params.robotTarget, simIK.constraint_pose)
    local ikJoints={}
    for i=1,6,1 do
        ikJoints[i] = simToIk[params.joints[i]]
    end
    sim.setObjectPose(params.robotTarget, pose)
    simIK.syncFromSim(ikEnv, {ikGroup}) -- make sure the arm is in the same configuration in the IK world!
    local p = {}
    p.findMultiple = true
    p.maxTime = 2.0

    local retVal = simIK.findConfigs(ikEnv, ikGroup, ikJoints, p)
    simIK.eraseEnvironment(ikEnv)
    return retVal
end

function collides(configs)
    -- checks if the configs are related to a collision with the environment or self-collision
    local retVal = false
    local bufferedConfig = getCurrConfig()
    for i = 1, #configs do
        setConfig(configs[i])

        -- Collision detection with small threshold
        local res, collidingObjs = sim.checkCollision(params.robotCollection, sim.handle_all)
        if res > 0 then
            retVal = true
            break
        else
            local res, collidingObjs = sim.checkCollision(params.robotCollection, params.robotCollection)
            if res > 0 then
                retVal = true
                break
            end
        end
    end
    
    setConfig(bufferedConfig)
    return retVal
end

function selectOneValidConfig(configs)
    local retVal, passiveVizShape
    for i = 1, #configs do
        local target = configs[i]
        if not collides({target}) then
            if target then
                retVal = target
                passiveVizShape = createPassiveShape(target)
                break
            end
        end
    end
    return retVal, passiveVizShape
end

-- OMPL functions   ------------------------------------------------------------------------------------------

function findPath(config)
    local useForProjection = {1,1,1,1,1,1}
    local result
    local task = simOMPL.createTask('path_task')
    simOMPL.setAlgorithm(task, params.pathPlanningAlgo)
    simOMPL.setStateSpaceForJoints(task, params.joints, useForProjection)
    simOMPL.setCollisionPairs(task, {params.robotCollection, sim.handle_all, params.robotCollection, params.robotCollection})
    simOMPL.setStateValidityCheckingResolution(task, params.pathPlanningResolution)
    simOMPL.setStartState(task, getCurrConfig())
    simOMPL.setGoalState(task, config)
    simOMPL.setup(task)

    if simOMPL.solve(task, params.pathPlanningMaxTime) and simOMPL.hasExactSolution(task) then
        simOMPL.simplifyPath(task, params.pathPlanningMaxSimplificationTime)
        simOMPL.interpolatePath(task, params.pathNStates)
        result = simOMPL.getPath(task)
    end

    simOMPL.destroyTask(task)
    
    return result
end

function getPath(pose)
    -- Move to pose
    local configs = findConfigs(pose)
    if #configs > 0 then
        print(string.format('Found %i different configs corresponding to the desired pick pose. Now selecting an appropriate valid config...', #configs))
        local pickConfig, passiveVizShape = selectOneValidConfig(configs)

        if not pickConfig then
            print('No valid configuration was found')
            return false
        end

        print('Selected following pick config: ', (Vector(pickConfig) * 180.0 / math.pi):data())
        
        local path = findPath(pickConfig)
        if path then
            print('Found a path from the current config to the pick config!')
            return path, passiveVizShape
        else
            print('Failed finding a path from the current config to the pick config. Try increasing the search times.')
        end 
    else
        print('Failed finding a config corresponding to the desired pick pose.')
    end
    return false
end

-- SuctionPad functions   ------------------------------------------------------------------------------------------

function detectSuctionSensor()
    local result, distance, point, object, n = sim.handleProximitySensor(params.suctionSensor)
    if result> 0 then
        print(string.format('SuctionPad sensor detected: %s', sim.getObjectAlias(object)))
        return object
    end
    return false
end

function toggleSuction(detectedObject, state)
    if not state then
        sim.setObjectParent(detectedObject, params.suction, true)
        sim.addItemToCollection(params.robotCollection, sim.handle_tree, detectedObject, 1)
        print(string.format('Object %s attached to suction pad',sim.getObjectAlias(detectedObject)))
    
    else
        sim.setObjectParent(detectedObject, -1, true)
        sim.setBoolProperty(detectedObject, 'dynamic', true)    -- Make object dynamic to fall
        sim.setBoolProperty(detectedObject, 'dynamic', true)
        sim.resetDynamicObject(detectedObject)
        print(string.format('Object %s detached to suction pad', sim.getObjectAlias(detectedObject)))
    end
end

-- System functions   ------------------------------------------------------------------------------------------

function sysCall_init()
    sim = require'sim'
    simIK = require'simIK'
    simOMPL = require'simOMPL'
    socket = require'socket'
    
    sim.setStepping(true)

    -- Prepare params for the arm
    params = {}
    params.joints = {}
    for i = 1, 6 do
        params.joints[i] = sim.getObject('/UR5/joint'..i)
    end
    params.robotTip=sim.getObject('/UR5/suctionPad/tip')
    params.robotTarget=sim.getObject('/UR5/target')
    params.robotBase=sim.getObject('/UR5')
    params.suction=sim.getObject('/UR5/suctionPad')
    params.suctionBody=sim.getObject('/UR5/suctionPad/Body')
    params.suctionSensor=sim.getObject('/UR5/suctionPad/Sensor')
    params.robotLink1=sim.getObject('/UR5/link1_visible')
    params.robotLink7=sim.getObject('/UR5/link7_visible')
    params.collisionBox=sim.getObject('/UR5/suctionPad/collisionBox')

    -- Make sure items arent dynamic
    local tunaCan=sim.getObject('/tunaCan')
    sim.setBoolProperty(tunaCan, 'dynamic', false)
    sim.resetDynamicObject(tunaCan)
    local clamp=sim.getObject('/clamp')
    sim.setBoolProperty(clamp, 'dynamic', false)
    sim.resetDynamicObject(clamp)
    local masterCan=sim.getObject('/masterCan')
    sim.setBoolProperty(masterCan, 'dynamic', false)
    sim.resetDynamicObject(masterCan)
    local sugarBox=sim.getObject('/sugarBox')
    sim.setBoolProperty(sugarBox, 'dynamic', false)
    sim.resetDynamicObject(sugarBox)

    -- Create robot arm collection
    params.robotCollection = sim.createCollection()
    sim.addItemToCollection(params.robotCollection, sim.handle_tree, params.robotBase, 0)

    params.pathPlanningMaxTime = 10.0
    params.pathPlanningMaxSimplificationTime = 2.0
    params.pathPlanningResolution = 0.01
    params.pathNStates = 20
    params.pathPlanningAlgo = simOMPL.Algorithm.RRTConnect
    params.downOri = {0, 0, -math.pi/2}

    -- IK motions:
    params.ikMaxVel = {0.4, 0.4, 0.4, 1.8}
    params.ikMaxAccel = {0.8, 0.8, 0.8, 0.9}
    params.ikMaxJerk = {0.6, 0.6, 0.6, 0.8}

    -- FK motions:
    local fkVel = 120
    local fkAccel = 40
    local fkJerk = 60
    params.fkMaxVel = {fkVel * math.pi / 180, fkVel * math.pi / 180, fkVel * math.pi / 180, fkVel * math.pi / 180, fkVel * math.pi / 180, fkVel * math.pi / 180}
    params.fkMaxAccel = {fkAccel * math.pi / 180, fkAccel * math.pi / 180, fkAccel * math.pi / 180, fkAccel * math.pi / 180, fkAccel * math.pi / 180, fkAccel * math.pi / 180}
    params.fkMaxJerk = {fkJerk * math.pi / 180, fkJerk * math.pi / 180, fkJerk * math.pi / 180, fkJerk * math.pi / 180, fkJerk * math.pi / 180, fkJerk * math.pi / 180}

    -- Home
    params.homePose = sim.getObjectPose(params.robotTip, -1)
    params.homeConfig = getCurrConfig()
    params.maxHeight = params.homePose[3]

    params.downOriQuat = eulerToQuaternion(params.downOri)
    params.heightDiff = 0.05

    params.movementDuration = 5.0

end

function getParams()
    return params    
end


function visualizePath(path, numSamples)
    assert(numSamples > 0, "Number of samples must be greater than 0")

    local numJoints = #params.joints -- Number of joints in the robot
    assert(#path % numJoints == 0, "Path length is not a multiple of the number of joints")

    -- Reshape the path into configurations
    local configs = {}
    for i = 1, #path, numJoints do
        local config = {}
        for j = 0, numJoints - 1 do
            table.insert(config, path[i + j])
        end
        table.insert(configs, config)
    end

    -- Determine the sampling step
    local step = math.max(1, math.floor(#configs / numSamples))
    local shapes = {}

    -- Create passive shapes for sampled configurations
    for i = 1, #configs, step do
        local config = configs[i]
        local shape = createPassiveShape(config)
        table.insert(shapes, shape)
    end

    print(string.format("Visualized the path with %d passive shapes", #shapes))
    return shapes
end


function findHomeTargetPath(location)

    -- Find all configs
    local configs = findConfigs(location)
    local validConf, passive = selectOneValidConfig(configs)

    if validConf == nil then
        print('no valid conf found for ', location)
    end

    local path = findPath(validConf)
    if path == nil then
        print('no valid path was found for ', location)
    end

    local p = {}
    p.path = path
    p.config = validConf

    sim.removeObjects({passive})

    return p
end

function initialParams(b) 
    if b then
        params.pathPlanningMaxTime = 20.0
        params.pathPlanningMaxSimplificationTime = 3.0
        params.pathPlanningResolution = 0.01
        params.pathNStates = 80
    
    else
        params.pathPlanningMaxTime = 10.0
        params.pathPlanningMaxSimplificationTime = 2.0
        params.pathPlanningResolution = 0.01
        params.pathNStates = 20
    end

    toggleCollisionBox(b)

end

function toggleCollisionBox(b)
    sim.setBoolProperty(params.collisionBox, 'collidable', b)
    sim.setBoolProperty(params.collisionBox, 'measurable', b)
    sim.setBoolProperty(params.collisionBox, 'detectable', b)
end