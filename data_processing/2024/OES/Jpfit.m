% Jpfit: function to be minimized for Langmuir probe fitting routine lang99c
function f = Jpfit(solV,Jfac,Vpv,Jp,iSheathExp,iFitEsat)

Npoints = length(Vpv);

Jsat = solV(1); Te = solV(2); Vs = solV(3); eslope = solV(4);
Vesat = solV(5); JslopeFit = solV(6);

if (iSheathExp == 1) % allow slope in Jsat
    Jfit = (Jsat + Vpv*JslopeFit).*(1 - Jfac*exp((Vpv-Vs)/Te));      
else
    Jfit = Jsat*(1 - Jfac*exp((Vpv-Vs)/Te));   
end

if (iFitEsat == 1)   % include esat correction
    Jesat = Jsat*(1 - Jfac*exp((Vesat-Vs)/Te));
    for ii = 1:Npoints
      if (Vpv(ii) >= Vesat)	% electron saturation current region
        Jfit(ii) = Jesat + eslope*(Vpv(ii)-Vesat);	% allow some arbitrary slope here
      end
    end
end

chi = Jfit - Jp;				% error between model and data

if (iFitEsat == 1)
    error = sum(abs(chi));
else  % if not fitting esat, focus on isat region for fit
   Jsort = sort(Jp);
   Jsatguess = abs(sum(Jsort(1:10))/10);  % quick guess for Jsat from data
   indexV = find(Jp<=Jsatguess); 
   error = sum(abs(chi(indexV))); % fit only points with Jp < abs(Jsat)
end

% turn up error alot for unphysical solutions                                
if ((Te <= 0.1) | ((abs(Jsat)) <= 1e-10) | (abs(Vs) > 100) ...
        | (eslope < 0) | (JslopeFit < 0) )
   error = error*100;	
end

f = error;
