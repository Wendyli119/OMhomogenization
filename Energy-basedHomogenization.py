# -*- coding: utf-8 -*-
"""
Numerical implementation of energy-based homogenization on Miura origami
Written by: Xuwen Li, Amin Jamalimehr

Please run the following codes in Abaqus/CAE using File > Run script and select this python file,
or run in ABAQUS command by typing: abaqus cae noGUI=Energy-basedHomogenization.py
Results are recorded in the text file effectiveConstantsEH.txt
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
from itertools import combinations_with_replacement
import itertools
import numpy as np
import math

foldAngles = np.array([30.])#
creaseStiffnesses = np.array([1.98])#
laminaPBC = {}
strain = ['e11','e22','e12','k11','k22','k12']
Job_Types = []
for (n1,n2) in list(combinations_with_replacement(strain, 2)):
    Job_Types.append(n1+n2)

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

        m = mdb.Model(name = modelName)
        a = m.rootAssembly
        ## Material properties
        m.Material(name='Mylar')
        m.materials['Mylar'].Elastic(table=((4000.0, 0.38), ))
        m.HomogeneousShellSection(name='Section1',
                                  material='Mylar',
                                  thickness=0.13)
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
        m.FieldOutputRequest(name='F-Output-3', createStepName='Step-1', variables=('ELSE',))
        
        # Locate midpoints of the creases
        creases = [[1,5],[4,5],[5,9],[5,6],[2,6],[6,7],[6,10],[8,9],[9,13],[9,10],[10,11],[10,14]]
        midpoints = []
        sprList = []
        connSet = []
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
        #Find all nodes on the boundaries
        FrontN = [3,7,11,15]
        BackN = [0,4,8,12]
        TopN = [12,13,14,15]
        BottomN = [0,1,2,3]
        BoundaryN = FrontN+BackN+TopN+BottomN
        connList = []
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
        
        # Locate a point not coincident to any hinges
        randpt = [0.,0.,-20.]
        # Assign torsional springs
        csyDict = {} 
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
                if csyName in csyDict.keys():
                    csy_loc = a.features[csyName]
                else:
                    csy_loc = a.DatumCsysByThreePoints(origin=ori, point1=xaxis, point2=xyplane, name=csyName, 
                                                    coordSysType=CARTESIAN)
                    csyDict[csyName] = csy_loc.id
                # Create a connector for each pair of nodes
                for ni in range(node_num):
                    # Check if a connector is needed
                    assign = 1
                    # Avoid conflict with PBC
                    for n in eval(setName):
                        if n in BoundaryN:
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
                                            localCsys1=a.datums[csyDict[csyName]])
        
        #Find all edges on the boundaries
        boundaries = []
        for panel in connect:
            for n in range(4):
                edge = [connect[panel][n],connect[panel][n+1]]
                for c in creases:
                    if set(edge)!=set(c):
                        boundaries.append(edge)
        connSet = []
        # Define connector properties
        jox = m.ConnectorSection(name='RJY', translationalType=JOIN, rotationalType=REVOLUTE)
        joint_e = jStiffness # Elasticity constant
        ekx = connectorBehavior.ConnectorElasticity(components=(4, ), 
                                                    table=((joint_e, ), )) # Define rotational stiffness on x-axis rotation (local coord system)
        jox.setValues(behaviorOptions =(ekx,) )
        for c in crConn.keys():
              user_joint(c,'RJY',randpt) 
        connRgn = a.SetByBoolean(name='Connectors', sets=connSet)
        m.FieldOutputRequest(name='F-Output-2', 
            createStepName='Step-1', variables=('CTF', 'CEF', 'CU', 'CUE', 'CUP'), 
            region=connRgn, sectionPoints=DEFAULT, rebar=EXCLUDE)
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
                            # Exclude corner nodes
                            coordn = np.array(n.coordinates)
                            if np.linalg.norm(coordn[0:2]-np.array([x[0],y[0]])) < panelSideLength/200:
                                nodeLoc = 'BottomBack'
                                Ri = regionToolset.Region(nodes=a.instances[n.instanceName].nodes.sequenceFromLabels(labels=(n.label,)))
                                a.Set(nodes=a.instances[n.instanceName].nodes.sequenceFromLabels(labels=(n.label,)), name=nodeLoc)
                            elif np.linalg.norm(coordn[0:2]-np.array([x[3],y[3]])) < panelSideLength/200:
                                nodeLoc = 'BottomFront'
                                Ri = regionToolset.Region(nodes=a.instances[n.instanceName].nodes.sequenceFromLabels(labels=(n.label,)))
                                a.Set(nodes=a.instances[n.instanceName].nodes.sequenceFromLabels(labels=(n.label,)), name=nodeLoc)
                            elif np.linalg.norm(coordn[0:2]-np.array([x[12],y[12]])) < panelSideLength/200:
                                nodeLoc = 'TopBack'
                                Ri = regionToolset.Region(nodes=a.instances[n.instanceName].nodes.sequenceFromLabels(labels=(n.label,)))
                                a.Set(nodes=a.instances[n.instanceName].nodes.sequenceFromLabels(labels=(n.label,)), name=nodeLoc)
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
        # Find the TopFront corner
        for n in a.sets['nodes'].nodes:
            coordn = np.array(n.coordinates)
            if np.linalg.norm(coordn[0:2]-np.array([x[15],y[15]])) < panelSideLength/200:
                nodeLoc = 'TopFront'
                Ri = regionToolset.Region(nodes=a.instances[n.instanceName].nodes.sequenceFromLabels(labels=(n.label,)))
                a.Set(nodes=a.instances[n.instanceName].nodes.sequenceFromLabels(labels=(n.label,)), name=nodeLoc)
        
        ## Equations (periodic boundary conditions)
        for (strain1,strain2) in list(combinations_with_replacement(strain, 2)):
            Type = strain1 + strain2
            for bc in m.boundaryConditions.keys():
                del m.boundaryConditions[bc]
            for pbc in m.constraints.keys():
                del m.constraints[pbc]
            cornerNodes = []
            nodeLocList = ['BottomBack','BottomFront','TopBack','TopFront']
            
            # PBC on edges
            origin = a.sets['BottomBack'].nodes[0].coordinates
            for boundary in ['Back','Bottom']:
                num_nodes = len(Boundaries[boundary]['nodes'])
                for n in range(num_nodes):
                    n1 = Boundaries[boundary]['nodes'][n]
                    n2 = Boundaries[Boundaries[boundary]['oppBoundary']]['nodes'][n]
                    i = 0
                    for nodeLoc in ['BottomFront','TopBack','TopFront']:
                        if n2 in a.sets[nodeLoc].nodes:
                            n1 = a.sets['BottomBack'].nodes[0]
                            i = 1
                    if i == 0: 
                        a.Set(nodes=a.instances[n1.instanceName].nodes.sequenceFromLabels(labels=(n1.label,)), name=str(n1.instanceName)+'N'+str(n1.label))
                        a.Set(nodes=a.instances[n2.instanceName].nodes.sequenceFromLabels(labels=(n2.label,)), name=str(n2.instanceName)+'N'+str(n2.label))
                        x_nb = n1.coordinates[0]-origin[0]
                        y_nb = n1.coordinates[1]-origin[1]
                        z_nb = n1.coordinates[2]
                        x_nf = n2.coordinates[0]-origin[0]
                        y_nf = n2.coordinates[1]-origin[1]
                        z_nf = n2.coordinates[2]
                        strainBC = {'e11':np.array([x_nf-x_nb,0,0]),'e22':np.array([0,y_nf-y_nb,0]),'e12':np.array([(y_nf-y_nb),(x_nf-x_nb),0]),
                                    'k11':np.array([(x_nf-x_nb)*((MaxZ+MinZ)/2.0-z_nf),0,(x_nf**2-x_nb**2)/2.0]),
                                    'k22':np.array([0,(y_nf-y_nb)*((MaxZ+MinZ)/2.0-z_nf),(y_nf**2-y_nb**2)/2.0]),
                                    'k12':np.array([(y_nf-y_nb)*((MaxZ+MinZ)/2.0-z_nf),(x_nf-x_nb)*((MaxZ+MinZ)/2.0-z_nf),(x_nf*y_nf-x_nb*y_nb)])}
                        if strain1 == strain2:
                            PBCutheta = strainBC[strain1]
                        else:
                            PBCutheta = strainBC[strain1]+strainBC[strain2]
                        rp = a.ReferencePoint(point=(3*S+x_nf, y_nf, z_nf))
                        rpName = 'RP'+n2.instanceName+str(n2.label)
                        rpSet = a.Set(name=rpName, referencePoints=(a.referencePoints[rp.id],))
                        tol = panelSideLength/200.0
                        
                        m.DisplacementBC(name=rpName,createStepName='Step-1',
                                          region=rpSet,u1=PBCutheta[0],u2=PBCutheta[1],u3=PBCutheta[2])
                    
                        for dof in DOF:
                            m.Equation(name=str(n1.instanceName)+'N'+str(n1.label)+str(n2.instanceName)+'N'+str(n2.label)+'DOF'+str(dof),
                                            terms=((-1.0,str(n1.instanceName)+'N'+str(n1.label),dof),
                                                  (1.0,str(n2.instanceName)+'N'+str(n2.label),dof),
                                                  (-1.0,rpName,dof)))
            # BC
            m.DisplacementBC(name='BottomBack',createStepName='Step-1',
                              region=a.sets['BottomBack'],u1=0,u2=0,u3=0,ur2=0) 
            ## Job
            job = mdb.Job(name='MiuraPBC-'+Type+modelid,model=modelName,numCpus=4, 
                          numDomains=4, numGPUs=0)
            try:
                job.submit()
                job.waitForCompletion()
            except:
                pass
        
        ## Postprocessing
        A = np.zeros((6,6))
        Auc = 2*S*2*L*H
        strainV = np.array([1,1,2,1,1,2])
        # Diagonals
        for i in range(6):
            Type = strain[i]
            JobName = 'MiuraPBC-'+Type+Type+modelid+'.odb'
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
            JobName = 'MiuraPBC-'+Type+modelid+'.odb'
            Etemp = session.openOdb(name=JobName).steps['Step-1'].frames[-1].fieldOutputs['ELSE']
            Euc = 0
            for val in Etemp.values:
                Euc = Euc + val.data
            strainV2 = np.zeros((1,len(strainV)))
            strainV2[0,i[0]] = 1
            strainV2[0,i[1]] = 1
            strainV2 = strainV2*strainV
            A[i[0], i[1]] = 2*Euc/Auc - np.dot(np.dot(strainV2, A2), np.transpose(strainV2))
            A[i[1],i[0]] = A[i[0],i[1]]
            session.odbs[JobName].close()
        Adiv = 2*np.outer(strainV, strainV)
        np.fill_diagonal(Adiv, 1)
        A = A/Adiv
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
        laminaPBC[modelid] = lamina
outtxt = open('effectiveConstantsEH.txt', 'w+')
outtxt.write(str(laminaPBC))
outtxt.close()