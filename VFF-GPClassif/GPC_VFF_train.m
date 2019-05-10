function model = GPC_VFF_train(X,y,options)
% X: (nxd) matrix of features for the training dataset. Each row is an instance.
% y: vector with the labels (0-1) for the training dataset.
% options: struct to indicate the algorithm parameters. It contains the fields:
%    - maxiter: Maximum number of iterations (in case there is not
%       convergence). Default = 200.
%    - thr: Threshold for the convergence of the hyperparameters theta(1)
%       and theta(2) of the code (which are \gamma and \sigma in the paper).
%       Default = 0,005.
%    - thr_angle: Threshold for the Fourier frequencies (the angle mean
%       difference is computed, thus we use an angular threshold different to
%       'thr'). Default = 1.
%    - D (or W, see lines 25-31 of the code): Number of Fourier Features (or explicit (Dxd) matrix of Fourier frequencies). 
%    - maxiter_par: Maximum number of iterations for conjugate gradient
%       maximization in eq.(11) of the paper. Default = 10.
%    - ker: kernel to be used ('rbf' for Gaussian Kernel is the only available option for the moment).
%    - iterOut: minimum number of consecutive iterations during which the
%       parameters must leave their initial conditions (in the sense of
%       keeping a distance greater than 'thr' with the previous value during
%       'iterOut' consecutive iterations). This is to avoid the method getting
%       stuck in the initial values. Default = 3.
%% Defining variables and initializing parameters
[n,d] = size(X);
if(isfield(options,'W'))
    W = options.W;
    D = size(W,1);
else
    D = options.D;
    W = randn(D,d);
end
v = (y-0.5);
s = estimateSigma(X,[],'mean');
W = W./s.mean;
theta = [1;1];
fprintf('%i \t %.7f \t %.7f ||| %.7f \t %.7f \t %.7f \t %.7f \n',[0,double(0),double(0),theta(1),theta(2),mean(W(:)),std(W(:))]);
%% Main Loop
stuck = true;
counter = 0;
for iter = 1:options.maxiter
    theta_old = theta;
    W_old = W;
    %% Generating the Random Fourier Features
    XW = X*W';
    Z = zeros(n,2*D);
    Z(:,1:2:(2*D)) = (theta(1)/realsqrt(D))*cos(XW./theta(2));
    Z(:,2:2:(2*D)) = (theta(1)/realsqrt(D))*sin(XW./theta(2));
    %% Updating xi
    if iter == 1
        xi = realsqrt((Z*(Z'*v)).^2+dot(Z,Z,2));
    else
        L = chol(Z'*bsxfun(@times,2*g_xi,Z)+eye(2*D),'lower');
        aux1 = L\(Z');
        xi = sqrt( (aux1'*(aux1*v)).^2 + dot(aux1,aux1)' );
    end
    g_xi = (1./(2*xi)) .* ((1./(1+exp(-xi))) - 0.5);
    %% Updating theta and W
    thetaLogW = minimize([log(theta);W(:)],@minus_log_prob,-options.maxiter_par,X,v,g_xi);
    theta = exp(thetaLogW(1:2));
    W = reshape(thetaLogW(3:end),D,d);
    if(theta(1)>1e6)
        theta(1) = 1e6;
        warning(['Iteration ',num2str(iter),': theta(1) greater than 1e6, re-adjusted to that value']);
    end
    if(theta(2)>1e4)
        theta(2) = 1e4;
        warning(['Iteration ',num2str(iter),': theta(2) greater than 1e4, re-adjusted to that value']);
    end
    isNull = (sum(W.^2,2)==0);
    W(isNull,:) = 1e-5*rand(sum(isNull),d);
    if(sum(isNull)>0)
        warning(['Iteration ',num2str(iter),': some null Fourier frequency, re-adjusted']);
    end
    %% Check convergence
    theta_c = norm(theta-theta_old)/norm(theta_old);
    W_c = (180/pi)*mean(acos(sum(W.*W_old,1)./(sqrt(sum(W.^2,1)).*sqrt(sum(W_old.^2,1)))));
    fprintf('%i \t %.7f \t %.7f ||| %.7f \t %.7f \t %.7f \t %.7f \n',[iter,theta_c,W_c,theta(1),theta(2),mean(W(:)),std(W(:))]);
    if (W_c < options.thr_angle && theta_c< options.thr)
        if(~stuck)
            fprintf('Convergence reached at iteration %i.\n',iter)
            break;
        else
            counter = 0;
        end
    else
        counter = counter + 1;
        if(counter==options.iterOut)
            stuck = false;
        end
    end
end
%% Generating the final (necessary components of) posterior distribution over \beta
XW = X*W';
Z = zeros(n,2*D);
Z(:,1:2:(2*D)) = (theta(1)/realsqrt(D))*cos(XW./theta(2));
Z(:,2:2:(2*D)) = (theta(1)/realsqrt(D))*sin(XW./theta(2));
L = chol(Z'*bsxfun(@times,2*g_xi,Z)+eye(2*D),'lower');
mu_beta = L'\(L\(Z'*v));
%% Outputs
model.W = W;
model.theta = theta;
model.xi = xi;
model.L = L;
model.mu_beta = mu_beta;
model.sigma_beta = diag(L'\(L\eye(2*D)));
% Only for the experiment
model.Z = Z;
end

% We take 'log(theta)' as parameter, because 'theta' itself must be
% positive and there could be problems with the 'minimize'
% Function to be minimized in eq.(11) of the paper
function [f,df] = minus_log_prob(thetaLogW,X,v,g_xi)
[n d] = size(X);
theta = exp(thetaLogW(1:2));
W = reshape(thetaLogW(3:end),[],d);
D = size(W,1);
XW = X*W';
Z = zeros(n,2*D);
Z(:,1:2:(2*D)) = (theta(1)/realsqrt(D))*cos(XW./theta(2));
Z(:,2:2:(2*D)) = (theta(1)/realsqrt(D))*sin(XW./theta(2));

%% Calculating some needed expressions
L = chol(Z'*bsxfun(@times,2*g_xi,Z)+eye(2*D),'lower');
aux1 = L\(Z'*v);                         %  L^{-1}Z'v
aux2 = L'\aux1;                          % (K+I)^{-1}Z'v
aux4 = 2*g_xi.*(Z*aux2);                 % (2A)Z(K+I)^{-1}Z'v
aux3 = (bsxfun(@times,2*g_xi,Z)/(L'))/L; % (2A)Z(K+I)^{-1}
%% Calculating function
f = -aux1'*aux1+2*sum(reallog(diag(L)));
%% Calculating gradient for theta
df = zeros(size(thetaLogW));
df(1) = 2*(aux1'*aux1)-2*sum(aux4.*aux4./(2*g_xi))-2*(aux3(:)'*Z(:));
df(2) = 0;
%% Calculating gradient for W
dfdW = zeros(size(W));
for i = 1:D
    dZdWij_gen = [-Z(:,2*i),Z(:,2*i-1)]./theta(2);
    for j = 1:d
        dZdWij = dZdWij_gen .* [X(:,j),X(:,j)];
        dfdW(i,j) = 2*v'*dZdWij*aux2((2*i-1):(2*i))-...
            2*aux4'*dZdWij*aux2((2*i-1):(2*i))-...
            2*sum(sum(aux3(:,(2*i-1):(2*i)).*dZdWij));
    end
end
%% Output
df(3:end) = dfdW(:);
df = -df;
end

function [X, fX, i] = minimize(X, f, length, varargin)

INT = 0.1;    % don't reevaluate within 0.1 of the limit of the current bracket
EXT = 3.0;                  % extrapolate maximum 3 times the current step-size
MAX = 20;                         % max 20 function evaluations per line search
RATIO = 10;                                       % maximum allowed slope ratio
SIG = 0.1; RHO = SIG/2; 
if max(size(length)) == 2, red=length(2); length=length(1); else red=1; end
if length>0, S='Linesearch'; else S='Function evaluation'; end 

i = 0;                                            % zero the run length counter
ls_failed = 0;                             % no previous line search has failed
[f0 df0] = feval(f, X, varargin{:});          % get function value and gradient
Z = X; X = unwrap(X); df0 = unwrap(df0);
%fprintf('%s %6i;  Value %4.6e\r', S, i, f0);
if exist('fflush','builtin') fflush(stdout); end
fX = f0;
i = i + (length<0);                                            % count epochs?!
s = -df0; d0 = -s'*s;           % initial search direction (steepest) and slope
x3 = red/(1-d0);                                  % initial step is red/(|s|+1)

while i < abs(length)                                      % while not finished
  i = i + (length>0);                                      % count iterations?!

  X0 = X; F0 = f0; dF0 = df0;                   % make a copy of current values
  if length>0, M = MAX; else M = min(MAX, -length-i); end

  while 1                             % keep extrapolating as long as necessary
    x2 = 0; f2 = f0; d2 = d0; f3 = f0; df3 = df0;
    success = 0;
    while ~success && M > 0
      try
        M = M - 1; i = i + (length<0);                         % count epochs?!
        
        [f3 df3] = feval(f, rewrap(Z,X+x3*s), varargin{:});
        df3 = unwrap(df3);
        if isnan(f3) || isinf(f3) || any(isnan(df3)+isinf(df3)), error(' '),end
        success = 1;
      catch                                % catch any error which occured in f
        x3 = (x2+x3)/2;                                  % bisect and try again
      end
    end
    if f3 < F0, X0 = X+x3*s; F0 = f3; dF0 = df3; end         % keep best values
    d3 = df3'*s;                                                    % new slope
    if d3 > SIG*d0 || f3 > f0+x3*RHO*d0 || M == 0  % are we done extrapolating?
      break
    end
    x1 = x2; f1 = f2; d1 = d2;                        % move point 2 to point 1
    x2 = x3; f2 = f3; d2 = d3;                        % move point 3 to point 2
    A = 6*(f1-f2)+3*(d2+d1)*(x2-x1);                 % make cubic extrapolation
    B = 3*(f2-f1)-(2*d1+d2)*(x2-x1);
    x3 = x1-d1*(x2-x1)^2/(B+sqrt(B*B-A*d1*(x2-x1))); % num. error possible, ok!
    if ~isreal(x3) || isnan(x3) || isinf(x3) || x3 < 0 % num prob | wrong sign?
      x3 = x2*EXT;                                 % extrapolate maximum amount
    elseif x3 > x2*EXT                  % new point beyond extrapolation limit?
      x3 = x2*EXT;                                 % extrapolate maximum amount
    elseif x3 < x2+INT*(x2-x1)         % new point too close to previous point?
      x3 = x2+INT*(x2-x1);
    end
  end                                                       % end extrapolation

  while (abs(d3) > -SIG*d0 || f3 > f0+x3*RHO*d0) && M > 0  % keep interpolating
    if d3 > 0 || f3 > f0+x3*RHO*d0                         % choose subinterval
      x4 = x3; f4 = f3; d4 = d3;                      % move point 3 to point 4
    else
      x2 = x3; f2 = f3; d2 = d3;                      % move point 3 to point 2
    end
    if f4 > f0           
      x3 = x2-(0.5*d2*(x4-x2)^2)/(f4-f2-d2*(x4-x2));  % quadratic interpolation
    else
      A = 6*(f2-f4)/(x4-x2)+3*(d4+d2);                    % cubic interpolation
      B = 3*(f4-f2)-(2*d2+d4)*(x4-x2);
      x3 = x2+(sqrt(B*B-A*d2*(x4-x2)^2)-B)/A;        % num. error possible, ok!
    end
    if isnan(x3) || isinf(x3)
      x3 = (x2+x4)/2;               % if we had a numerical problem then bisect
    end
    x3 = max(min(x3, x4-INT*(x4-x2)),x2+INT*(x4-x2));  % don't accept too close
    [f3 df3] = feval(f, rewrap(Z,X+x3*s), varargin{:});
    df3 = unwrap(df3);
    if f3 < F0, X0 = X+x3*s; F0 = f3; dF0 = df3; end         % keep best values
    M = M - 1; i = i + (length<0);                             % count epochs?!
    d3 = df3'*s;                                                    % new slope
  end                                                       % end interpolation

  if abs(d3) < -SIG*d0 && f3 < f0+x3*RHO*d0          % if line search succeeded
    X = X+x3*s; f0 = f3; fX = [fX' f0]';                     % update variables
    %fprintf('%s %6i;  Value %4.6e\r', S, i, f0);
    if exist('fflush','builtin') fflush(stdout); end
    s = (df3'*df3-df0'*df3)/(df0'*df0)*s - df3;   % Polack-Ribiere CG direction
    df0 = df3;                                               % swap derivatives
    d3 = d0; d0 = df0'*s;
    if d0 > 0                                      % new slope must be negative
      s = -df0; d0 = -s'*s;                  % otherwise use steepest direction
    end
    x3 = x3 * min(RATIO, d3/(d0-realmin));          % slope ratio but max RATIO
    ls_failed = 0;                              % this line search did not fail
  else
    X = X0; f0 = F0; df0 = dF0;                     % restore best point so far
    if ls_failed || i > abs(length)         % line search failed twice in a row
      break;                             % or we ran out of time, so we give up
    end
    s = -df0; d0 = -s'*s;                                        % try steepest
    x3 = 1/(1-d0);                     
    ls_failed = 1;                                    % this line search failed
  end
end
X = rewrap(Z,X); 
%fprintf('\n'); 
if exist('fflush','builtin') fflush(stdout); end
end

function v = unwrap(s)
% Extract the numerical values from "s" into the column vector "v". The
% variable "s" can be of any type, including struct and cell array.
% Non-numerical elements are ignored. See also the reverse rewrap.m. 
v = [];   
if isnumeric(s)
  v = s(:);                        % numeric values are recast to column vector
elseif isstruct(s)
  v = unwrap(struct2cell(orderfields(s))); % alphabetize, conv to cell, recurse
elseif iscell(s)
  for i = 1:numel(s)             % cell array elements are handled sequentially
    v = [v; unwrap(s{i})];
  end
end                                                   % other types are ignored
end
function [s v] = rewrap(s, v)
% Map the numerical elements in the vector "v" onto the variables "s" which can
% be of any type. The number of numerical elements must match; on exit "v"
% should be empty. Non-numerical entries are just copied. See also unwrap.m.
if isnumeric(s)
  if numel(v) < numel(s)
    error('The vector for conversion contains too few elements')
  end
  s = reshape(v(1:numel(s)), size(s));            % numeric values are reshaped
  v = v(numel(s)+1:end);                        % remaining arguments passed on
elseif isstruct(s) 
  [s p] = orderfields(s); p(p) = 1:numel(p);      % alphabetize, store ordering
  [t v] = rewrap(struct2cell(s), v);                 % convert to cell, recurse
  s = orderfields(cell2struct(t,fieldnames(s),1),p);  % conv to struct, reorder
elseif iscell(s)
  for i = 1:numel(s)             % cell array elements are handled sequentially 
    [s{i} v] = rewrap(s{i}, v);
  end
end
end

function [sigma cost] = estimateSigma(X,Y,method)

% Subsampling
[n d] = size(X);
idx = randperm(n);
if n>1000
    n = 1000;
    X = X(idx(1:n),:);
    if ~isempty(Y)
        Y = Y(idx(1:n),:);
    end
end

% Range of sigmas
ss = 20;
SIGMAS = logspace(-3,3,ss);

if sum(strcmpi(method,'mean'))
    t=cputime;
    D = pdist(X);
    sigma.mean = mean(D(D>0));
    cost.mean = cputime-t;
end
end