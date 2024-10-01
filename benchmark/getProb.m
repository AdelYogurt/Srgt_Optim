function varargout=getProb(name,vn)
% obtain benchmark function by benchmark type and name
%
if nargin < 2
    vn=[];
    if nargin < 1
        name=[];
    end
end

self_filestr=which('getProb');
[self_filedir]=fileparts(self_filestr);

% get problem name list
file_list=[
    dir(fullfile(self_filedir,'sunc*.m'));
    dir(fullfile(self_filedir,'scon*.m'));
    dir(fullfile(self_filedir,'munc*.m'));
    dir(fullfile(self_filedir,'mcon*.m'));];
prob_str=[file_list.name];
prob_list={file_list.name};

prob_str=strrep(prob_str,'sunc','');
prob_str=strrep(prob_str,'scon','');
prob_str=strrep(prob_str,'munc','');
prob_str=strrep(prob_str,'mcon','');
prob_str=strrep(prob_str,'.m',' ');
prob_str=split(prob_str,{' ','_'});
dat_list=reshape(prob_str(1:end-1),2,[])';

vn_list=str2double(dat_list(:,1));
name_list=dat_list(:,2);

if isempty(name)
    varargout={vn_list,name_list};
else
    prob_idx=find(strcmp(name,name_list));

    if isnan(vn_list(prob_idx))
        prob=prob_list{prob_idx};
        prob=strrep(prob,'.m','');
        [obj_fcn,vari_num,A,b,Aeq,beq,low_bou,up_bou,nonlcon_fcn,x_best,obj_best]=eval(prob,vn);
    else
        prob=prob_list{prob_idx};
        prob=strrep(prob,'.m','');
        [obj_fcn,vari_num,A,b,Aeq,beq,low_bou,up_bou,nonlcon_fcn,x_best,obj_best]=eval(prob);
    end

    con_fcn=@(x)conFcn(x,A,b,Aeq,beq,nonlcon_fcn);
    objcon_fcn=@(x)objconFcn(x,obj_fcn,con_fcn);

    varargout={objcon_fcn,vari_num,low_bou,up_bou,x_best,obj_best};
end
end


