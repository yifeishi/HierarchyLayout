function output = demoMatrix(affinity_dir,hier_dir);
disp(affinity_dir)
disp(hier_dir)
fileID = fopen(affinity_dir,'r');
outputID = fopen(hier_dir, 'w');
data = fscanf(fileID, '%f');
fclose(fileID);

fileID = fopen(affinity_dir,'r');
leafNum=0;
while ~feof(fileID)
        str=fgetl(fileID);
    if ~isempty(str)
        leafNum=leafNum+1;
    end
end
fclose(fileID);

W = reshape(data,leafNum,[]);% yifei

for i=1:leafNum
    for j=1:leafNum
        %if W(i,j) < 0.01
        %    W(i,j) = rand(1)*2;
        %else
        %    W(i,j) = W(i,j) * 100;
        %end
        W(i,j) = W(i,j);
    end
end

%disp(W);

nbCluster = 2;
rootNum = leafNum*2-2;
valid=ones(rootNum,1);
valid(rootNum)=0;
fprintf('leafNum %d  rootNum %d\n',leafNum, rootNum)

function [id,valid] = findNextID(valid)
    id = length(valid);
    for i = length(valid):-1:0
        if valid(i)==1
            id = i;
            valid(id)=0;
            break
        end
    end
end

%%left childs, right childs, weight matrix
function [valid] = buildTree(curRoot, rootList, valid)
    if size(rootList,2) == 3
        [id,valid] = findNextID(valid);
        fprintf(outputID, '%d %d null\n', curRoot, id);   
        fprintf(outputID, '%d %d %d\n', curRoot, rootList(3)-1, rootList(3)-1);
        
        fprintf(outputID, '%d %d %d\n', id, rootList(1)-1, rootList(1)-1);
        fprintf(outputID, '%d %d %d\n', id, rootList(2)-1, rootList(2)-1);
        return;
    elseif size(rootList,2) == 2
        fprintf(outputID, '%d %d %d\n', curRoot, rootList(1)-1, rootList(1)-1);
        fprintf(outputID, '%d %d %d\n', curRoot, rootList(2)-1, rootList(2)-1);
        return;
    end
    
    compute = false;
    
    try
        [NcutDiscrete,~,~] = ncutW(W(rootList,rootList),nbCluster);
    catch
        lx = length(rootList);
        half = ceil(lx/2);
        lcs = rootList(1:half);
        rcs = rootList(half+1:end);
        lcount = size(lcs, 2);
        rcount = size(rcs, 2);
        compute = true;
        if lcount > 10
            fprintf('equally split left: %d  right: %d\n',lcount, rcount)
        end
    end
    
    if ~compute
        for j=1:nbCluster
            id = find(NcutDiscrete(:,j));
            actualId = rootList(id);
            if j==1
               lcount = size(actualId, 2);
               lcs = actualId;
            else
               rcount = size(actualId, 2);
               rcs = actualId;
            end
        end
        %fprintf('left: %d  right: %d\n',lcount, rcount)
    end

    if lcount == 0 || rcount == 0
        fprintf('zero left: %d  right: %d\n',lcount, rcount)
        lx = length(rootList);
        half = ceil(lx/2);
        lcs = rootList(1:half);
        rcs = rootList(half+1:end);
        lcount = size(lcs, 2);
        rcount = size(rcs, 2);
        fprintf('corrected left: %d  right: %d\n',lcount, rcount)
    end
    
    if lcount > 1 && rcount > 1
        [leftID,valid] = findNextID(valid);
        fprintf(outputID, '%d %d null\n', curRoot, leftID);
        [rightID,valid] = findNextID(valid);
        fprintf(outputID, '%d %d null\n', curRoot, rightID);
        valid = buildTree(leftID, lcs, valid);
        valid = buildTree(rightID, rcs, valid);
    elseif lcount == 1 && rcount == 1
        fprintf(outputID, '%d %d %d\n', curRoot, lcs(1)-1, lcs(1)-1);
        fprintf(outputID, '%d %d %d\n', curRoot, rcs(1)-1, rcs(1)-1);
    elseif lcount == 1 && rcount ~= 1
        fprintf(outputID, '%d %d %d\n', curRoot, lcs(1)-1, lcs(1)-1);
        [rightID,valid] = findNextID(valid);
        fprintf(outputID, '%d %d null\n', curRoot, rightID);
        valid = buildTree(rightID, rcs, valid);
    elseif lcount ~= 1 && rcount == 1
        fprintf(outputID, '%d %d %d\n', curRoot, rcs(1)-1, rcs(1)-1);
        [leftID,valid] = findNextID(valid);
        fprintf(outputID, '%d %d null\n', curRoot, leftID);
        valid = buildTree(leftID, lcs, valid);
    else

    end

end

buildTree(rootNum, (1:leafNum), valid);% yifei
output = 1;
%disp(rootNum);
end

