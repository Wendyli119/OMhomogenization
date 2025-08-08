# -*- coding: utf-8 -*-
"""
Numerical implementation of asymptotic homogenization on Miura origami
Written by: Xuwen Li, Amin Jamalimehr

Please run the following codes in Abaqus/CAE using File > Run script and select this python file,
or run in ABAQUS command by typing: abaqus cae noGUI=asymptoticHomogenization.py
Results are recorded in the text file effectiveConstantsAH.txt
"""

from abaqus import *
import testUtils
testUtils.setBackwardCompatibility()
from abaqusConstants import *
from caeModules import *
import sketch
import part
from regionToolset import*
import load
from shutil import *
from odbAccess import *
import assembly
from itertools import chain
from itertools import combinations_with_replacement
import itertools
import numpy as np
import math

tol = 1e-4
foldAngles = np.array([30.])#
creaseStiffnesses = np.array([1.98])#
strain = ['e11','e22','e12','k11','k22','k12']#
Job_Types = []
for (n1,n2) in list(combinations_with_replacement(strain, 2)):
    Job_Types.append(n1+n2)
laminaAH = {}
DOF = [1,2,3]
for angle in foldAngles:
    for Kcr in creaseStiffnesses:
        modelName = 'thetaKcr'
        modelid = 'theta'+str(int(angle))+'Kcr'+str(Kcr).replace(".","_")
        ## Draw a sketch for each part
        theta=np.pi*angle/180 #fold angle
        gmma = 60*np.pi/180 #sector angle
        zeta = np.arctan(np.cos(theta)*np.tan(gmma))
        psi = np.arcsin(np.sin(theta)*np.sin(gmma))
        panelSideLength = 20.0
        a = 20.0
        b = 20.0
        meshSize = 1.0/16.0*a
        H = a*np.sin(theta)*np.sin(gmma)
        S = b*np.cos(theta)*np.tan(gmma)/np.sqrt(1+np.cos(theta)**2*np.tan(gmma)**2)
        L = a*np.sqrt(1-np.sin(gmma)**2*np.sin(theta)**2)
        V = b/np.sqrt(1+np.cos(theta)**2*np.tan(gmma)**2)
        x = np.array([-S/2.0, 0, S, 3.0/2.0*S, -S/2.0, 0, S, 3.0/2.0*S,-S/2.0, 0, S, 3.0/2.0*S, -S/2.0, 0, S, 3.0/2.0*S,
                      -S/2.0, 0, S, 3.0/2.0*S, -S/2.0, 0, S, 3.0/2.0*S,-S/2.0, 0, S, 3.0/2.0*S, -S/2.0, 0, S, 3.0/2.0*S])
        y = np.array([-L/2.0+V/2.0, -L/2.0, -L/2.0+V, -L/2.0+V/2.0, 
                      V/2.0, 0, V, V/2.0, 
                      L+V/2.0, L, L+V, L+V/2.0,
                      3.0/2.0*L+V/2.0, 3.0/2.0*L, 3.0/2.0*L+V, 3.0/2.0*L+V/2.0,
                      -L/2.0+V/2.0, -L/2.0, -L/2.0+V, -L/2.0+V/2.0, 
                        V/2.0, 0, V, V/2.0, 
                        L+V/2.0, L, L+V, L+V/2.0,
                        3.0/2.0*L+V/2.0, 3.0/2.0*L, 3.0/2.0*L+V, 3.0/2.0*L+V/2.0])
        z = np.array([H/2.0, H/2.0, H/2.0, H/2.0, 0, 0, 0, 0, H, H, H, H, H/2.0, H/2.0, H/2.0, H/2.0,
                      3.0/2.0*H, 3.0/2.0*H, 3.0/2.0*H, 3.0/2.0*H, 2.0*H, 2.0*H, 2.0*H, 2.0*H, H, H, H, H, 3.0/2.0*H, 3.0/2.0*H, 3.0/2.0*H, 3.0/2.0*H])
        connect = {0:[0, 1, 5, 4, 0], # Node connectivity
                    1:[1, 2, 6, 5, 1],
                    2:[2, 3, 7, 6, 2],
                    3:[6, 7, 11, 10, 6],
                    4:[5, 6, 10, 9, 5],
                    5:[4, 5, 9, 8, 4],
                    6:[8, 9, 13, 12, 8],
                    7:[9, 10, 14, 13, 9],
                    8:[10, 11, 15, 14, 10]
                    }
        creases = [[1,5],[4,5],[5,9],[5,6],[2,6],[6,7],[6,10],[8,9],[9,13],[9,10],[10,11],[10,14]]
        #Find all nodes on the boundaries
        FrontN = [3,7,11,15]
        BackN = [0,4,8,12]
        TopN = [12,13,14,15]
        BottomN = [0,1,2,3]
        
        ## Assign connectors
        vertices = {5:[],
                    6:[],
                    9:[],
                    10:[]}
        for vertex in vertices: 
            for panel in connect.keys():
                if vertex in connect[panel]:
                    vertices[vertex].append(panel)
            vertices[vertex].sort()# a list of panels connected to the vertex
        # Put boundary mesh nodes into groups
        Boundaries = {'Back':{'nodes':[],'midpoints':[],'vertices':BackN,
                              'oppBoundary':'Front','latticeVector':np.array([2*S,0,0])},
                      'Bottom':{'nodes':[],'midpoints':[],'vertices':BottomN,
                                'oppBoundary':'Top','latticeVector':np.array([0,2*L,0])}}
        groupedBoundaries = Boundaries.keys()
        ## Interaction: crease property assignment
        # Define crease stiffness
        # Number of node per crease
        nnpc = int(b/meshSize + 1)
        # Stiffness of the crease averaged over each node
        jStiffness = np.divide(Kcr, nnpc)
        # Append filenames with the crease stiffness
        HStiffness = modelid
        # Low stiffness for the unit strain displacement field
        UjStiffness = np.divide(1e-7, nnpc)
        UHStiffness = modelid

        def buildUC():        
            m = mdb.Model(name = modelName)
            a = m.rootAssembly
            ## Material properties
            m.Material(name='Mylar')
            m.materials['Mylar'].Elastic(table=((4.0e3, 0.38), ))
            m.HomogeneousShellSection(name='Section1',
                                      material='Mylar',
                                      thickness=0.13)
            
            edgeList = []
            for panel in connect.keys():
                p = m.Part(name='Panel'+str(panel), dimensionality=THREE_D, 
                        type=DEFORMABLE_BODY)
                p.ReferencePoint(point=(0.0, 0.0, 0.0))
                d1 = [0] * int(np.amax(np.array(connect.values()))+1) #list of datum points defining vertices
                for n in connect[panel][0:-1]:
                    d1[n]=p.DatumPointByCoordinate(coords=(x[n], y[n], z[n]))
                w = [0] * 4 #list of wires defining panel edges
                e=p.edges
                for n in range(len(connect[panel])-1):
                    n1 = connect[panel][n]
                    n2 = connect[panel][n+1]
                    w[n]=p.WirePolyLine(points=((p.datums[d1[n1].id], p.datums[d1[n2].id]),), mergeType=IMPRINT, meshable=ON)
                    edgeList.append([[n1,n2],panel])
                p.ShellLoft(loftsections=((e[0], ), (e[2], )), paths=((e[1], ), (e[3], )), 
                    globalSmoothing=ON)
                ## Section
                p.SectionAssignment(region=Region(faces=p.faces), sectionName='Section1')
                ## Instance
                ins = a.Instance(name='panel_'+str(panel), part=p, dependent=OFF)
                ## Mesh
                elem_type = mesh.ElemType(elemCode=S4R)
                a.setElementType(regions=(ins.faces,), elemTypes=(elem_type,))
                a.seedEdgeBySize(edges=ins.edges, constraint=FIXED, size=meshSize)
                a.setMeshControls(regions=ins.faces, elemShape=TRI, allowMapped=ON, technique=FREE)
                a.generateMesh(regions=(ins,))
        
            ## Step
            m.StaticStep(name='Step-1', previous='Initial', maxNumInc=100000, 
                initialInc=0.001, maxInc=0.1)
            m.FieldOutputRequest(name='F-Output-3', createStepName='Step-1', variables=('E', 'COORD' ))
            
            # Locate midpoints of the creases
            midpoints = []
            crConn = {} #dictionary of creases with their connecting panels
            for c in creases:
                  midpoints.append([0.5*(x[c[0]]+x[c[1]]),0.5*(y[c[0]]+y[c[1]]),0.5*(z[c[0]]+z[c[1]]),c])
                  crConn[str(c)]=[]
            for mp in midpoints:
                  for panel in connect.keys():
                      e = a.instances['panel_'+str(panel)].edges.findAt((mp[0:3],),)
                      if len(e) > 0:
                          a.Set(edges=e, name=str(mp[3])+'panel_'+str(panel)) # Create a set for the crease passing mp
                          crConn[str(mp[3])].append(panel)
            for c in crConn.keys():
                crConn[c].sort() #sort the list of panels in ascending order
            # Define connector properties
            jox = m.ConnectorSection(name='RJY', translationalType=JOIN, rotationalType=REVOLUTE)
            joint_e = jStiffness # Elasticity constant
            ekx = connectorBehavior.ConnectorElasticity(components=(4, ), 
                                                        table=((joint_e, ), )) # Define rotational stiffness on x-axis rotation (local coord system)
            jox.setValues(behaviorOptions =(ekx,) )
            
            ## Boundary conditions: unit strain displacement fields
            # Create a node list
            nodeList = [] # n is the variable for all nodes in the unit cell
            nodeLabels = [] # Node labels of all nodes
            nseq = []
            for ins in a.instances.keys():
                for i in a.instances[ins].nodes:
                    nodeList.append(i)
                    nodeLabels.append((i.label,ins))
                nseq.append(a.instances[ins].nodes)
            N = len(nodeList) # Total number of nodes
            # Create a set for all nodes
            a.Set(name='nodes',nodes=tuple(nseq))
            
            return [m, a, nodeList, nodeLabels, N, crConn, edgeList]
        
        ##
        # Find nodal reactions of unit strain displacement fields
        ##
        # A function to assign displacements
        # Distance calculator
        def Distance(P,Q):
        
            Px = P[0]
            Qx = Q[0]
            
            Py = P[1]
            Qy = Q[1]
        
            Pz = P[2]
            Qz = Q[2]
        
            dist = math.sqrt((Px-Qx)**2 + (Py-Qy)**2 + (Pz-Qz)**2)
        
            return(dist)
        # Corresponding node detection: returns indices of nodes2 that match the original order of nodes1
        def corresponding_detector(nodes1,nodes2):
            length_nodes1 = len(nodes1)
            length_nodes2 = len(nodes2)
            record = [0] * length_nodes1
            D = np.zeros(length_nodes1)
        
            for i in range(0,length_nodes1):
        
                coordinates_nodes1 = nodes1[i][0:3]
        
                for j in range(0,length_nodes2):
        
                    coordinates_nodes2 = nodes2[j][0:3]
        
                    D[j] = Distance(coordinates_nodes1,coordinates_nodes2)
        
                record[i] = D.argmin() # Index of the matching node in nodes2
                
            return(record)
        # A function to assign torsional springs between corresponding nodes on coincident edges
        def springs(setName,xyplane):
            for panelANum in range(len(crConn[setName])-1):
                panelAstr = 'panel_'+str(crConn[setName][panelANum])
                panelA = crConn[setName][panelANum]
                panelBstr = 'panel_'+str(crConn[setName][panelANum+1])
                panelB = crConn[setName][panelANum+1]
                # Nodes on edges a and b
                nodes_a = a.sets[setName+panelAstr].nodes
                nodes_b = a.sets[setName+panelBstr].nodes
                node_num = len(nodes_a)
            
                # Coordinates of the nodes
                coord_a = np.zeros((node_num,4))
                coord_b = np.zeros((node_num,4))
                
                # Record node coordinates and label. Nodes with the same label are not necessarily coincident.
                for ni in range (0,node_num):
                    coord_a[ni][0:3] = nodes_a[ni].coordinates[0:3]
                    coord_a[ni][3] = ni     
                    coord_b[ni][0:3] = nodes_b[ni].coordinates[0:3]
                    coord_b[ni][3] = ni
                record = corresponding_detector(coord_a,coord_b)
                coord_b = [coord_b[i] for i in record]
                # Create a local coordinate system (x axis along hinge)
                ori = coord_a[0][0:3]
                xaxis = coord_a[-1][0:3]
                csyName = 'csy_local'+setName+str(UjStiffness).replace(".","_")
                if csyName in csyDict.keys():
                    csy_loc = a.features[csyName]
                else:
                    csy_loc = a.DatumCsysByThreePoints(origin=ori, point1=xaxis, point2=xyplane, name=csyName, 
                                                    coordSysType=CARTESIAN)
                    csyDict[csyName] = csy_loc.id
                # Create a torsional spring for each pair of nodes
                for ni in range (node_num):
                    # Check if a connector is needed
                    assign = 1
                    # Avoid conflict with PBC
                    for n in eval(setName):
                        for boundary in groupedBoundaries:
                            if n in Boundaries[boundary]['vertices']:
                                bdist = Distance([x[n],y[n],z[n]],nodes_a[ni].coordinates[0:3]) 
                                if bdist < 0.01:
                                    assign = 0
                    #Avoid conflict at the vertices
                    for vertex in vertices:
                        if vertex in eval(setName):
                            vdist = Distance([x[vertex],y[vertex],z[vertex]],nodes_a[ni].coordinates[0:3])
                            if vdist < 0.01:
                                if vertices[vertex][0] in set([panelA,panelB]) and vertices[vertex][-1] in set([panelA,panelB]):
                                    #skip the 4th connector assignment to avoid conflict
                                    assign = 0
                    if assign == 0: # No need for a connector
                        continue
                    if assign == 1: # assign a connector to the node pair
                        springName = setName + str(ni) + panelAstr + panelBstr
                        id_a   = int(coord_a[ni][3]) # Node on edge a
                        id_b   = int(coord_b[ni][3]) # Corresponding node on edge b
                        sprList.append([nodes_a[id_a].instanceName+'N'+str(nodes_a[id_a].label), nodes_b[id_b].instanceName+'N'+str(nodes_b[id_b].label)])
                    rgn1pair0=regionToolset.Region(nodes=nodes_a[id_a:(id_a+1)])
                    rgn2pair0=regionToolset.Region(nodes=nodes_b[id_b:(id_b+1)])
                    region=((rgn1pair0, rgn2pair0), )
                    datum = a.datums[csy_loc.id]
                    a.engineeringFeatures.TwoPointSpringDashpot(
                        name=springName, regionPairs=region, axis=FIXED_DOF, dof1=4, 
                        dof2=4, orientation=datum, springBehavior=ON, 
                        springStiffness=UjStiffness, dashpotBehavior=OFF, 
                        dashpotCoefficient=0.0)
        def disp_assigner(label,JobName,BCname):
            Translational_DOF = [1,2,3]
            Rotational_DOF    = [4,5,6]
            setName =  'C'+ Type + str(label[1]) + 'N' + str(label[0])
            BCRegion = regionToolset.Region(nodes=a.instances[label[1]].nodes.sequenceFromLabels(labels=(label[0],)))
            Value = np.zeros(6)
            if BCname=='CD':
                ufield = session.openOdb(name='Uniform-'+JobName+'.odb').steps['Step-1'].frames[-1].fieldOutputs
                usubset = session.odbs['Uniform-'+JobName+'.odb'].rootAssembly.instances[str(label[1]).upper()]
                cfield = session.openOdb(name='Characteristic-'+JobName+'.odb').steps['Step-1'].frames[-1].fieldOutputs
                csubset = session.odbs['Characteristic-'+JobName+'.odb'].rootAssembly.instances[str(label[1]).upper()]
                for dof in DOF:
                    if dof in Translational_DOF:
                        Value[dof-1] = ufield['U'].getSubset(region=usubset).values[label[0]-1].data[dof-1] + cfield['U'].getSubset(region=csubset).values[label[0]-1].data[dof-1]
                    if dof in Rotational_DOF:
                        Value[dof-1] = ufield['UR'].getSubset(region=usubset).values[label[0]-1].data[dof-4] + cfield['UR'].getSubset(region=csubset).values[label[0]-1].data[dof-4]
            else:
                field = session.openOdb(name='UnitStrain-'+JobName+'.odb').steps['Step-1'].frames[-1].fieldOutputs
                subset = session.odbs['UnitStrain-'+JobName+'.odb'].rootAssembly.instances[str(label[1]).upper()]
                for dof in DOF:
                    if dof in Translational_DOF:
                        Value[dof-1] = field['U'].getSubset(region=subset).values[label[0]-1].data[dof-1]
                    if dof in Rotational_DOF:
                        Value[dof-1] = field['UR'].getSubset(region=subset).values[label[0]-1].data[dof-4]
            m.DisplacementBC(name=BCname+Type+setName,
                          createStepName='Step-1',region=BCRegion,
                          u1=Value[0],u2=Value[1],u3=Value[2])#,ur1=Value[3],ur2=Value[4],ur3=Value[5]
        
        [m, a, nodeList, nodeLabels, N, crConn, edgeList] = buildUC()
        ## Boundary conditions: unit strain displacement fields
        csyDict = {}
        sprList = []
        UjStiffness = jStiffness
        # Locate a point not coincident to any hinges
        randpt = [0.,0.,-panelSideLength*2.0]
        for c in crConn.keys():
         	springs(c,randpt) 
        connSet = []
        sprList = [
        sprList1
        for sprList2 in sprList
        for sprList1 in sprList2]
        # Find the outer dimensions of the unit cell
        MinX = nodeList[0].coordinates[0]; MaxX = nodeList[0].coordinates[0];
        MinY = nodeList[0].coordinates[1]; MaxY = nodeList[0].coordinates[1];
        MinZ = nodeList[0].coordinates[2]; MaxZ = nodeList[0].coordinates[2];
        for i in range(0,N):
        
            c = nodeList[i].coordinates
        
            if c[0] < MinX:
                MinX = c[0]
            if c[0] > MaxX:
                MaxX = c[0]
        
            if c[1] < MinY:
                MinY = c[1]
            if c[1] > MaxY:
                MaxY = c[1]
        
            if c[2] < MinZ:
                MinZ = c[2]
            if c[2] > MaxZ:
                MaxZ = c[2]
        ## Compute the unit strain displacement fields
        for (strain1,strain2) in list(combinations_with_replacement(strain, 2)):
            Type = strain1 + strain2
            JobName = Type+modelid
            # supress all connectors and PBCs
            for bc in m.boundaryConditions.keys():
                del m.boundaryConditions[bc]
            # # assign unit strain displacement fields
            for i in nodeList:
                X = i.coordinates[0] - MinX
                Y = i.coordinates[1] - MinY
                Z = i.coordinates[2] - (MinZ+MaxZ)/2.0
                Ri = regionToolset.Region(nodes=a.instances[i.instanceName].nodes.sequenceFromLabels(labels=(i.label,)))
                strainBC = {'e11':np.array([float(X),0.0,0.0]),
                            'e22':np.array([0.0, float(Y),0.0]),
                            'e12':np.array([float(0.5*(Y)), float(0.5*(X)), 0.0]),
                            'k11':np.array([float(Z*X),0.0,float(-X**2/2.0)]),
                            'k22':np.array([0.0,float(Z*Y),float(-Y**2/2.0)]),
                            'k12':np.array([float(Z*Y/2.0),float(Z*X/2.0),float(-X*Y/2.0)])}
                if strain1 == strain2:
                    UnitStrainU = strainBC[strain1]
                else:
                    UnitStrainU = strainBC[strain1] + strainBC[strain2]
                m.DisplacementBC(name='US'+Type+str(i.label)+i.instanceName,
                                  createStepName='Step-1',region=Ri,
                                  u1=UnitStrainU[0],u2=UnitStrainU[1],u3=UnitStrainU[2])
            ## Job
            job = mdb.Job(name='Uniform-'+Type+modelid,model=modelName,numCpus=4, 
                          numDomains=4, numGPUs=0)
            try:
                job.submit()
                job.waitForCompletion()
            except:
                pass
        
        # A function to write the nodal displacements into a txt file
        def write_output(fields, N, Type, Category):
            # Displacement vectors
            resultName = Category+'-'+Type+'.txt'
            results = open(resultName, 'w')
            for i in range(N):
                results.write(str(i+1)) # Node number
                for j in range(3): # Node displacements Ux, Uy, Uz
                    results.write(' '+str(fields['U'].values[i].data[j]))
                for j in range(3): # Node rotations Rx, Ry, Rz
                    results.write(' '+str(fields['UR'].values[i].data[j]))
                results.write(' \n')
                
            subset = session.odbs[Category+'-'+Type+'.odb'].rootAssembly.instances['ASSEMBLY']
            for i in range(len(subset.elements)):
                if Category in ['Uniform','CharNF']:
                    results.write(str(i+1)) # Node number
                    results.write(' '+str(fields['E'].getSubset(region=subset).values[i].data[0]))
                if Category == 'Characteristic':
                    results.write(str(i+1)) # Node number
                    results.write(' '+str(fields['CUR'].getSubset(region=subset).values[i].data[0]))
                results.write(' \n')
                
            results.close()
            # Force vectors
            resultNameF = Category+'-'+Type+'F.txt'
            results = open(resultNameF, 'w')
            for i in range(N):
                results.write(str(i+1)) # Node number
                for j in range(3): # Nodal reaction forces Fx, Fy, Fz
                    results.write(' '+str(fields['RF'].values[i].data[j]))
                for j in range(3): # Nodal reaction moments Mx, My, Mz
                    results.write(' '+str(fields['RM'].values[i].data[j]))
                results.write(' \n')
                
            subset = session.odbs[Category+'-'+Type+'.odb'].rootAssembly.instances['ASSEMBLY']
            for i in range(len(subset.elements)):
                if Category in ['Uniform','CharNF']:
                    results.write(str(i+1)) # Node number
                    results.write(' '+str(fields['S'].getSubset(region=subset).values[i].data[0]))
                if Category == 'Characteristic':
                    results.write(str(i+1)) # Node number
                    results.write(' '+str(fields['CTM'].getSubset(region=subset).values[i].data[0]))
                results.write(' \n')
            results.close()
        # # Write the uniform X results into a txt file using the function
        fields = {}
        for Type in Job_Types:
            JobName = 'Uniform-'+Type+modelid+'.odb'
            fields[Type] = session.openOdb(name=JobName).steps['Step-1'].frames[-1].fieldOutputs # Results of the last frame in the first step
            
        ##
        # Find characteristic displacement fields
        ##
        # A function to assign connectors
        def user_joint(setName,joint_name,xyplane):#setName is the end nodes of the crease
            for panelANum in range(len(crConn[setName])-1):
                panelAstr = 'panel_'+str(crConn[setName][panelANum])
                panelA = crConn[setName][panelANum]
                panelBstr = 'panel_'+str(crConn[setName][panelANum+1])
                panelB = crConn[setName][panelANum+1]
                # Nodes on edges a and b
                nodes_a = a.sets[setName+panelAstr].nodes
                nodes_b = a.sets[setName+panelBstr].nodes
                node_num = len(nodes_a)
                
                # Coordinates of the nodes
                coord_a = np.zeros((node_num,4))
                coord_b = np.zeros((node_num,4))
                
                # Record node coordinates and label. Nodes with the same label are not necessarily coincident.
                for ni in range(0,node_num):
                    coord_a[ni][0:3] = nodes_a[ni].coordinates[0:3]
                    coord_a[ni][3] = ni     
                    coord_b[ni][0:3] = nodes_b[ni].coordinates[0:3]
                    coord_b[ni][3] = ni
                # Sort the nodes in b according to their distances from a
                record = corresponding_detector(coord_a,coord_b)
                coord_b = [coord_b[i] for i in record]
                # Create a local coordinate system (x axis along hinge)
                ori = coord_a[0][0:3]
                xaxis = coord_a[-1][0:3]
                csyName = 'csy_local'+setName+str(UjStiffness).replace(".","_")
                csy_loc = a.DatumCsysByThreePoints(origin=ori, point1=xaxis, point2=xyplane, name=csyName, 
                                                coordSysType=CARTESIAN)
                csyid = a.features['csy_local'+setName+str(jStiffness).replace(".","_")].id
                # Create a connector for each pair of nodes
                for ni in range(node_num):
                    # Check if a connector is needed
                    assign = 1
                    # Avoid conflict with PBC
                    for n in eval(setName):
                        for boundary in groupedBoundaries:
                            if n in Boundaries[boundary]['vertices']:
                                bdist = Distance([x[n],y[n],z[n]],coord_a[ni][0:3]) 
                                if bdist < 0.01:
                                    assign = 0
                    #Avoid conflict at the vertices
                    for vertex in vertices:
                        if vertex in eval(setName):
                            vdist = Distance([x[vertex],y[vertex],z[vertex]],coord_a[ni][0:3])
                            if vdist < 0.01:
                                if vertices[vertex][0] in set([panelA,panelB]) and vertices[vertex][-1] in set([panelA,panelB]):
                                    #skip the 4th connector assignment to avoid conflict
                                    assign = 0
                    if assign == 0: # No need for a connector
                        continue
                    if assign == 1: # assign a connector to the node pair
                        id_a   = int(coord_a[ni][3]) # Node on edge a
                        id_b   = int(coord_b[ni][3]) # Corresponding node on edge b
                        wir = a.WirePolyLine(points=((nodes_a[id_a],nodes_b[id_b]),), 
                                          meshable=OFF) # A wire that connects the two nodes on each edge
                        connList.append([nodes_a[id_a].instanceName+'N'+str(nodes_a[id_a].label), nodes_b[id_b].instanceName+'N'+str(nodes_b[id_b].label)])
                    
                    # Find the edge created by WirePolyLine() in the root assembly
                    for ei in range(len(a.edges)):
                        if a.edges[ei].featureName == wir.name:
                            break
                    # Create a set for the wire (actually its corresponing edge in the root assembly)
                    temp_name = setName+'-'+str(ni)+ panelAstr + panelBstr
                    connSet.append(a.Set(edges=a.edges[ei:(ei+1)], name=temp_name))
                    # Assign properties
                    a.SectionAssignment(sectionName=joint_name, 
                                        region=a.sets[temp_name])
                    a.ConnectorOrientation(region=a.sets[temp_name], 
                                            localCsys1=a.datums[csyid])
        
        [m, a, nodeList, nodeLabels, N, crConn, edgeList] = buildUC()
        #Find all edges on the boundaries
        boundaries = []
        for panel in connect:
            for n in range(4):
                edge = [connect[panel][n],connect[panel][n+1]]
                for c in creases:
                    if set(edge)!=set(c):
                        boundaries.append(edge)
        connSet = []
        connList = []
        # Define connector properties
        jox = m.ConnectorSection(name='RJY', translationalType=JOIN, rotationalType=REVOLUTE)
        joint_e = jStiffness # Elasticity constant
        ekx = connectorBehavior.ConnectorElasticity(components=(4, ), 
                                                    table=((joint_e, ), )) # Define rotational stiffness on x-axis rotation (local coord system)
        jox.setValues(behaviorOptions =(ekx,) )
        for c in crConn.keys():
              user_joint(c,'RJY',randpt) 
        file = open("Connectors.txt", "w")
        numConn = len(connList)
        for i in range(numConn):
            file.write(connList[i][0]+'\n')
            file.write(connList[i][1]+'\n')
            file.write('\n')
        file.close()
        connRgn = a.SetByBoolean(name='Connectors', sets=connSet)
        m.FieldOutputRequest(name='F-Output-2', 
            createStepName='Step-1', variables=('CTF', 'CEF', 'CU', 'CUE', 'CUP'), 
            region=connRgn, sectionPoints=DEFAULT, rebar=EXCLUDE)
        
        # Find the back and bottom nodes
        for boundary in Boundaries:
            for (n1,n2) in list(itertools.combinations(Boundaries[boundary]['vertices'], 2)):
                for e in edgeList:
                    if set(e[0]) == set([n1,n2]): #Check if [n1,n2] forms an edge
                        Boundaries[boundary]['midpoints'].append([0.5*(x[n1]+x[n2]),0.5*(y[n1]+y[n2]),0.5*(z[n1]+z[n2]),[n1,n2]])
                        break
            for mp in Boundaries[boundary]['midpoints']:
                for panel in connect.keys():
                    e = a.instances['panel_'+str(panel)].edges.findAt((mp[0:3],),)
                    if len(e) > 0:
                        a.Set(edges=e, name='edge' + str(mp[3][0])+'_'+str(mp[3][1]))
                        for n in a.sets['edge' + str(mp[3][0])+'_'+str(mp[3][1])].nodes:
                            Boundaries[boundary]['nodes'].append(n)
        # Find corresponding nodes in the opposite boundaries
        for boundary in groupedBoundaries:
            ob = Boundaries[boundary]['oppBoundary']
            Boundaries[ob] = {'nodes':[]}
            for n in Boundaries[boundary]['nodes']:
                oriCoord = np.array(n.coordinates)
                transCoord = oriCoord + Boundaries[boundary]['latticeVector']
                # Find the node at transCoord
                for edgeNode in nodeList:
                    c2 = np.array(edgeNode.coordinates)
                    if np.linalg.norm(transCoord-c2)<meshSize/4:
                        Boundaries[ob]['nodes'].append(edgeNode)
                        break
                    
        PBCedges = {}
        for boundaryCombinations in list(itertools.combinations(groupedBoundaries,2)):
            b1nodes = Boundaries[boundaryCombinations[0]]['nodes']
            b2nodes = Boundaries[boundaryCombinations[1]]['nodes']
            edgeName = (boundaryCombinations[0],boundaryCombinations[1])
            edgelist = [ [] for i in range(4)]
            for n1 in b1nodes:
                if n1 in b2nodes:
                    edgelist[0].append(n1)
                    c1 = []
                    c1.append(np.array(n1.coordinates))
                    lv1 = Boundaries[boundaryCombinations[0]]['latticeVector']
                    lv2 = Boundaries[boundaryCombinations[1]]['latticeVector']
                    c1.append(c1[0] + lv1)
                    c1.append(c1[0] + lv2)
                    c1.append(c1[0] + lv1 + lv2)
                    for i in range(1,len(c1)):
                        for n2 in nodeList:
                            c2 = np.array(n2.coordinates)
                            if np.linalg.norm(c1[i]-c2)<0.01:
                                edgelist[i].append(n2)
                                break
            PBCedges[edgeName]=edgelist           
        flat_PBCedges = [item2 for sublist in PBCedges.values() for item1 in sublist for item2 in item1]
        ## Equations (periodic boundary conditions)
        def write_equations(n1,n2):
            a.Set(nodes=a.instances[n1.instanceName].nodes.sequenceFromLabels(labels=(n1.label,)), name=str(n1.instanceName)+'N'+str(n1.label))
            a.Set(nodes=a.instances[n2.instanceName].nodes.sequenceFromLabels(labels=(n2.label,)), name=str(n2.instanceName)+'N'+str(n2.label))
            for dof in DOF:
                m.Equation(name=str(n1.instanceName)+'N'+str(n1.label)+str(n2.instanceName)+'N'+str(n2.label)+'DOF'+str(dof),
                            terms=((1.0,str(n1.instanceName)+'N'+str(n1.label),dof),
                                  (-1.0,str(n2.instanceName)+'N'+str(n2.label),dof)))
                
        # Interior nodes of boundary faces
        for boundary in groupedBoundaries:
            num_nodes = len(Boundaries[boundary]['nodes'])
            for n in range(num_nodes):
                n1 = Boundaries[boundary]['nodes'][n]
                n2 = Boundaries[Boundaries[boundary]['oppBoundary']]['nodes'][n]
                c1 = np.array(n1.coordinates)
                PBCedgeNode = 0
                for pbcedgenode in flat_PBCedges:
                    if n1 == pbcedgenode:
                        PBCedgeNode = 1
                if PBCedgeNode: # skip all boundary edges
                    continue
                write_equations(n1,n2)
        # Boundary edges
        for edgeName in PBCedges:
            if PBCedges[edgeName][0]:
                for n in range(len(PBCedges[edgeName][0])):
                    for edge in range(len(PBCedges[edgeName])-1):
                        n1 = PBCedges[edgeName][edge][n]
                        n2 = PBCedges[edgeName][edge+1][n]
                        write_equations(n1,n2)
        
        ## Boundary condition preventing rigid body motions
        for bc in m.boundaryConditions.keys():
            del m.boundaryConditions[bc]
        for label in nodeLabels:
            Set = str(label[1])+'N'+str(label[0])
            if Set in list(chain.from_iterable(connList)): # Skip assigning BC if a node is at a connector
                continue
            elif Set in a.sets.keys(): # Skip assigning BC if a node is at the unit cell boundary
                continue
            else:
                break
        fixedBC = regionToolset.Region(nodes=a.instances[label[1]].nodes.sequenceFromLabels(labels=(label[0],)))
        m.EncastreBC(name='fixedBC', createStepName='Initial', 
            region=fixedBC, localCsys=None)
        
        ## Loads: apply reaction forces from the unit strain displacement field
        # A function to assign loads
        def load_assigner(label,field):
            Translational_DOF = [1,2,3]
            Rotational_DOF    = [4,5,6]
            setName =  'C'+ Type + str(label[1]) + 'N' + str(label[0])
            loadRegion = regionToolset.Region(nodes=a.instances[label[1]].nodes.sequenceFromLabels(labels=(label[0],)))
            Value = np.zeros(6)
            for dof in DOF:
                
                if dof in Translational_DOF:
                    subset = session.odbs['Uniform-'+Type+modelid+'.odb'].rootAssembly.instances[str(label[1]).upper()]
                    Value[dof-1] = - field['RF'].getSubset(region=subset).values[label[0]-1].data[dof-1]
                    
                if dof in Rotational_DOF:
                    subset = session.odbs['Uniform-'+Type+modelid+'.odb'].rootAssembly.instances[str(label[1]).upper()]
                    Value[dof-1] = - field['RM'].getSubset(region=subset).values[label[0]-1].data[dof-4]
                
            if np.linalg.norm(Value[0:3]) > tol:
                m.ConcentratedForce(name=setName+'cf', createStepName='Step-1', 
                                        region=loadRegion, cf1=Value[0], cf2=Value[1], cf3=Value[2], distributionType=UNIFORM, field='', 
                                        localCsys=None)
            
        for Type in Job_Types:
            for cf in m.loads.keys():
                m.loads[cf].suppress() 
            for label in nodeLabels:
                load_assigner(label,fields[Type])
            ## Job
            job = mdb.Job(name='Characteristic-'+Type+modelid,model=modelName,numCpus=4, 
                          numDomains=4, numGPUs=0)
            try:
                job.submit()
                job.waitForCompletion()
            except:
                pass  
        
        ##
        # Find difference between uniform and characteristic displacement fields
        ##
        [m, a, nodeList, nodeLabels, N, crConn, edgeList] = buildUC()
        csyDict = {}
        for c in crConn.keys():
         	springs(c,randpt) 
        m.FieldOutputRequest(name='F-Output-else', createStepName='Step-1', variables=('ELSE',))
        for Type in Job_Types:
            JobName = Type+modelid
            # assign displacement field
            for bc in m.boundaryConditions.keys():
                m.boundaryConditions[bc].suppress()
            for label in nodeLabels:
                disp_assigner(label,JobName,'CD')
            ## Job
            job = mdb.Job(name='Diff-'+Type+modelid,model=modelName,numCpus=4, 
                          numDomains=4, numGPUs=0)
            try:
                job.submit()
                job.waitForCompletion()
            except:
                pass
        
        ## Postprocessing
        A = np.zeros((6,6))
        Auc = 2*S*2*L*H
        strainV = np.array([1,1,1,1,1,1])
        # Diagonals
        for i in range(6):
            Type = strain[i]
            JobName = 'Diff-'+Type+Type+modelid+'.odb'
            Etemp = session.openOdb(name=JobName).steps['Step-1'].frames[-1].fieldOutputs['ELSE']
            Euc = 0
            for val in Etemp.values:
                Euc = Euc + val.data
            A[i,i] = 2*Euc/Auc/strainV[i]**2
            session.odbs[JobName].close()
        A2 = A
        # Off-diagonal terms
        for i in list(itertools.combinations(range(6),2)):
            Type = strain[i[0]]+strain[i[1]]
            JobName = 'Diff-'+Type+modelid+'.odb'
            Etemp = session.openOdb(name=JobName).steps['Step-1'].frames[-1].fieldOutputs['ELSE']
            Euc = 0
            for val in Etemp.values:
                Euc = Euc + val.data
            strainV2 = np.zeros((1,len(strainV)))
            strainV2[0,i[0]] = 1
            strainV2[0,i[1]] = 1
            strainV2 = strainV2*strainV
            A[i[0], i[1]] = Euc/Auc - 0.5*np.dot(np.dot(strainV2, A2), np.transpose(strainV2))
            A[i[1],i[0]] = A[i[0],i[1]]
            for ok in session.odbs.keys():
                session.odbs[ok].close()
        # Compute the compliance matrix
        SA = np.linalg.inv(A)

        # Elastic constants
        lamina = {}
        lamina['E1'] = 1.0/SA[0][0]
        lamina['E2'] = 1.0/SA[1][1]
        lamina['G12'] = 1.0/SA[2][2]
        lamina['v12'] = -SA[1][0]*lamina['E1']
        lamina['v21'] = -SA[0][1]*lamina['E2']
        lamina['M1'] = 1.0/SA[3][3]
        lamina['M2'] = 1.0/SA[4][4]
        lamina['T12'] = 1.0/SA[5][5]
        lamina['vb12'] = -SA[4][3]*lamina['M1']
        lamina['vb21'] = -SA[3][4]*lamina['M2']
        laminaAH[modelid] = lamina
        
outtxt = open('effectiveConstantsAH.txt', 'w+')
outtxt.write(str(laminaAH))
outtxt.close()