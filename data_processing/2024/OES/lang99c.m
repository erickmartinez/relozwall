% This program lang99c analyses slow-swept Langmuir probe data.
% Magnetized probe theory from Hutchinson's book is used.
% We include a provision for a finite space potential and for
% the ion mass to be different from unity.
% A maxwellian electron distribution function is used.
% lang99b is taken from lang99, but takes log(J+offset) rather than log(abs(J)).
% Also, only the horizontal information from the mouse input is used.
% lang99c improves on lang99b by allowing both manual fitting to the data as well as 
% numerically fitting over the whole probe sweep using the manual fits as an initial guess.
% lang99c requires data of the form dataV(:,1) = probe voltage signal[V]
%									dataV(:,2) = probe current signal[V]
%									dataV(:,3) = probe position signal[V]

% Pisces-A probe data
shot = 7; 
% shotp = 1; % shot with plunge data (can be different from main shot)
% mystring = sprintf('load gamma_ivdata%0.4i.raw',shotp);
% eval(mystring);
% mystring = sprintf('dataV(:,3) = gamma_ivdata%0.4i(:,3);',shotp); % position signal
% eval(mystring);
mystring = sprintf('load ''/Users/erickmartinez/Library/Mobile Documents/com~apple~CloudDocs/Documents/GitHub/relozwall/data_processing/2024/OES/data/PA_probe/20240815/gamma_ivdata%0.4i.raw''',shot);
eval(mystring);
mystring = sprintf('dataV(:,1) = gamma_ivdata%0.4i(:,1);',shot); % voltage signal
eval(mystring);
mystring = sprintf('dataV(:,2) = gamma_ivdata%0.4i(:,2);',shot); % current signal
eval(mystring);
mystring = sprintf('dataV(:,3) = gamma_ivdata%0.4i(:,3);',shot); % position signal
eval(mystring);
dataV(:,2) = -dataV(:,2); % flip current trace

% Pisces-B probe data
% shot = 19786;
% mystring = sprintf('load /Users/Eric/Data/PA_sputter_cone/20170718/probe/S20170718%i/data.raw',shot);
% eval(mystring);
% dataV(:,1) = data(:,1);
% dataV(:,2) = data(:,2);
% dataV(:,3) = data(:,3);

% ibin = 1;
% binning for high-density data sets
% if (ibin==1)
% dataVlong = dataV;
% clear dataV;
% binsize = 4;
% Nlong = length(dataVlong(:,1));
% Nshort = round(Nlong/binsize)-1;
% for ii = 1:Nshort
%    ilow = (ii-1)*binsize+1;
%    ihigh = ii*binsize;
%    dataV(ii,1) = sum(dataVlong(ilow:ihigh,1))/binsize;
%    dataV(ii,2) = -sum(dataVlong(ilow:ihigh,2))/binsize;
%    dataV(ii,3) = sum(dataVlong(ilow:ihigh,3))/binsize;
% end
% end

% ***************** Problem parameters **********************************

% Pisces-A probe geometry
% AreaP = .085;   % probe tip area [cm^2], corresponds to 4*R*L, two direction projected area
% AreaP = .025; % probe tip area [cm^2], corresponds to 2*R*L, single direction projected area for probe near target
AreaP = .049; % probe tip area [cm^2] that Daisuke is using at present (10/2024) for gamma probe
Vscale = 100;	% voltage division done in Langmuir probe box :  Vprobe = dataV(:,1)*Vscale
Rc = 20;        % current resistor [Ohms]in Langmuir probe box : Iprobe = dataV(:,2)/Rc. Usually 2 Ohm or 5 Ohm
% xScale = 1.56;	% probe plunge position conversion [cm/V]
xScale = 1.53; % probe plunge position conversion [cm/V] that Daisuke is using (10/2024)

% Pisces-B probe geometry
% AreaP = 0.013; % probe tip area [cm^2], corresponds to 2*R*L, just one direction projected area
% Vscale = 100;
% Rc = 20;
% xScale = 5.4; 

Amag = 0.61; 	% Use Amag = 0.5 for magnetized plasma (rci^2 << Ap) ; 0.61 for unmagnetized plasma
mi = 2;			% mi is ion mass in units of hydrogen mass. Typically use 4 for He+ and 2 for D+
Zion = 1;		% ion charge state
Tescale = 0.67; % scale factor on Te for estimating correct plasma potential on manual fits

iRampType = 3;	% set to 1 for rising sawtooth; 2 for triangle wave; 3 for sine wave
iLeak = 1;          % set to 1 to correct for leakage current
iSheathExp = 1;  % allow empirical correction for sheath expansion factor in numerical fit
iFitEsat = 1;   % set to 1 if interested in fitting esat; otherwise just fit isat region
Ti = 0.5;         % ion temperature
IpAtten = 1; % attenuation on Ip signal from isolator

ee = 1.6e-19;	% electron charge. 
rpe = 1836.2;	% proton over electron mass ratio
Jfac = sqrt(rpe*mi/(2*pi))/(2*Amag);

Vpv = dataV(:,1)*Vscale;	% convert signal measured after probe box to actual probe V,J
Jp = IpAtten*dataV(:,2)/(AreaP*Rc);	% probe current density [A/cm^2]
xV = dataV(:,3)*xScale;		% position vector [cm]


% ******************* Characterize probe voltage sawtooth ******************

Npoints = length(Vpv);						% total number of data points in sawtooth
VpvOff = Vpv - sum(Vpv)/Npoints;			% center Vpv vertically to remove k=0 Fourier component
fVp = fft(VpvOff);							% Fourier transform sawtooth
[km,im] = max(abs(fVp(1:round(Npoints/2))));	% locate largest peak in Fourier spectrum
lambda = round(Npoints/(im-1));					% wavelength of sawtooth [points]

if (iRampType == 1)
   Vpp = 6.34*abs(fVp(im))/Npoints;		% peak-peak amplitude of sawtooth [V]
else
    Vpp = max(Vpv(1:(2*lambda))) - min(Vpv(1:(2*lambda)));
	%Vpp = 5.8*abs(fVp(im))/Npoints;		% peak-peak amplitude of triangle ramp [V]
end
lambda0 = lambda*(pi/2 + atan(imag(fVp(im)))/(real(fVp(im))));	% phase of sawtooth [points]
lambda0 = round(lambda0);
if (lambda0 > lambda); lambda0 = lambda0 - lambda; end
iStart = lambda0;		% initialize search for first ramp at lambda0
if (iRampType > 1); [kp,iStart] = max(Vpv(1:lambda)); end
% here, define the phase as the distance to the bottom of the first complete sawtooth ramp [points]
% in the case of a triangle ramp, it's the distance to the top of the first "V"

Nramps = 0;				% initialize number of good ramps to 0
NrampsMx = Npoints/lambda;	% maximum number of ramps
for iramp = 1:NrampsMx
   ilo = iStart - 10;	% anticipate about 100 points per ramp, so search +/- 10 points around peak
   ihi = iStart + 10;
   if (ilo < 1); ilo = 1; end
   if (ihi > Npoints); ihi = Npoints; end
   ilo = round(ilo); ihi = round(ihi); 
   if (iRampType == 1)
        [dum,idum] = min(Vpv(ilo:ihi));   % iStart is beginning of ramp "/" for sawtooth
    else                                                % or sine wave
       [dum,idum] = max(Vpv(ilo:ihi));    % iStart is beginning of "V" for triangle
   end
   iStart = ilo + idum - 1;
   iEnd = iStart + lambda;	                    % iEnd is end of ramp for sawtooth
   if (iRampType > 1) ; iEnd = iStart + lambda/2; end   % iEnd is bottom of "V" for triangle
   ilo = iEnd - 10;
   ihi = iEnd + 10;
   if (ilo < 1); ilo = 1; end
   if (ihi > Npoints); ihi = Npoints; end
   ilo = round(ilo); ihi = round(ihi);
   if (iRampType == 1)
        [dum,idum] = max(Vpv(ilo:ihi));	   
    else
        [dum,idum] = min(Vpv(ilo:ihi));
    end
   iEnd = ilo + idum - 1;
   iStart = round(iStart);
   iEnd = round(iEnd);
   VppMeas = Vpv(iEnd) - Vpv(iStart);	% amplitude of this sawtooth
   if (iRampType > 1); VppMeas = -VppMeas; end
   if ((VppMeas > 0.5*Vpp) & (VppMeas < 1.5*Vpp)) 		% check to make sure this ramp looks reasonable
      Nramps = Nramps + 1;
      iStartRamp(Nramps) = round(iStart);
      iEndRamp(Nramps) = round(iEnd);			% characterize ramps by their start and end indices
   end
   if (iEnd > (Npoints - lambda + 5)); break; end		% exit loop
   
   if (iRampType == 1)
        iStart = iEnd + 5;   % prepare loop for next ramp
    else
        iStart = iStart + lambda;
    end

end

% plot out ramp data and lines for each ramp 
figure
plot(Vpv,'o')
ylabel('V-probe')
title('Probe voltage and fit to ramps')
hold on
for iramp = 1:Nramps-1
   iStart = iStartRamp(iramp);
   iEnd = iEndRamp(iramp);
   if (iRampType == 1)
        plot([iStart iEnd],[Vpv(iStart) Vpv(iEnd)],'r-');
    elseif (iRampType == 2)
        plot([iStart iEnd],[Vpv(iStart) Vpv(iEnd)],'r-');
        plot([iEnd iStart+lambda],[Vpv(iEnd) Vpv(iStart+lambda)],'r-'); 
    else    % iRampType = 3
        Vpp = Vpv(iStart) - Vpv(iEnd);
        Voff = (Vpv(iStart) + Vpv(iEnd))/2;
        plot([iStart:iStart+lambda],Voff+0.5*Vpp*cos(([iStart:iStart+lambda]-iStart)*2*pi/lambda),'r-')
    end
end

% zero initial guesses for all vectors
TeMan(1:Nramps) = 1;		neMan(1:Nramps) = 0; 	VsMan(1:Nramps) = 0;  JsatMan(1:Nramps) = 0;
% vectors corresponding to Te, ne, and Vs obtained from manual fits to data
TeFit(1:Nramps) = 1;		neFit(1:Nramps) = 0;		VsFit(1:Nramps) = 0;  eslopeFit(1:Nramps) = 0;
VesatFit(1:Nramps) = 0;  esatoisat(1:Nramps) = 0; JsatFit(1:Nramps) = 0; JslopeFit(1:Nramps) = 0;
Jsat(1:Nramps) = 0; JsatR(1:Nramps) = 0; VsManC(1:Nramps) = 0;
% vectors corresponding to Te, ne, Vs, slope of J in eSat region, probe V at which eSat region begins, 
% esat to isat ratio, Jsat, and slope of Isat region of characteristic. JsatFit is initial fit; Jsat is JsatFit corrected
% for possible slope in JsatFit; JsatR is simple, robust fit to Jsat just averaging data in Isat region

VJsatRmax = 0.75*min(Vpv);    % for simple Jsat calc, assume we're in Jsat for V < VJsatRmax


% ************** Fit to leakage current **********************
if (iLeak == 1)
    disp(' ')
    sst = ['Data consists of ',num2str(Nramps),' voltage sweeps'];
    disp(sst)
    disp(' ')
    disp('Input indices of sweeps to be fit for leakage current');
    disp('These should be in a region with no plasma signal');
    iRampsLeak = input('[iStart,iEnd]: ');
%    iRampsLeak(1) = 1; iRampsLeak(2) = 3;    % just assume there's no plasma in first several sweeps
    clear Jpleak pfit
    Jpleak(1:lambda) = 0;    
    for iramp = iRampsLeak(1):iRampsLeak(2)     % step through each ramp
           iStart = iStartRamp(iramp);
           for ii = 1:lambda
               Jpleak(ii) = Jpleak(ii) + Jp(ii-1+iStart);
           end
    end
    Jpleak = Jpleak/(iRampsLeak(2)-iRampsLeak(1)+1);
    pfit = polyfit((1:lambda),Jpleak,25);
    Jpleakfit = polyval(pfit,(1:lambda));
    figure
    plot(Jpleak,'rx')
    hold on
    plot(Jpleakfit,'b--')               % plot out average data and fit
    title('Fit to leakage current')
    
    for iramp = 1:Nramps-1
        iStart = iStartRamp(iramp);
        iEnd = iStart+lambda-1;
        for ii = iStart:iEnd
            Jp(ii) = Jp(ii) - Jpleakfit(ii-iStart+1);       % subtract off leakage current signal
        end
    end
    
end

% **************** Manual fit to characteristic *******************

% choose which voltage ramps are to be fit manually
disp(' ')
sst = ['Data consists of ',num2str(Nramps),' voltage sweeps'];
disp(sst)
disp(' ')
iRampsMan = input('Input indices of sweeps to be fit manually in form [iStart,iEnd]: ');
% iRampsMan(1) = 25; iRampsMan(2) = 26;

% step through each ramp and do manual fit to characteristic
figure
imandone = 0;
iramp = iRampsMan(1);
while (imandone==0) % cycle through manual fits
   iStart = iStartRamp(iramp);
   iEnd = iEndRamp(iramp);
   if (iRampType > 1)
       iStart = iEndRamp(iramp);                        % manual fit to second leg of "V"
       iEnd = iStartRamp(iramp) + round(lambda);
   end
   Vpfit = Vpv(iStart:iEnd);
   Jfit = Jp(iStart:iEnd);
   
	Jmin = min(Jfit);			% offset probe current by 1.1*Jmin
	Jmin0 = 1.1*abs(Jmin); 

	JpOffset = Jfit + Jmin0;	% JpOffset is Jp offset to be positive-definite

	yjprobe = log(JpOffset);		% now compute log of probe current.

	disp(' ')
	disp(['Probe data for sweep ',num2str(iramp)])
	plot(Vpfit,yjprobe,'o')
	xlabel(['V-probe'])
    ylabel('logJ-probe')
    title('Mouse select range for Te slope finder','Color',[1,0,0],'FontSize',14)
	hold on
	grid
   
	% Get electron temperature from plot
	[Vrange,yJrange] = ginput(2);			% acccept graphical input from mouse

	Vmax = max(Vrange);
	Vmin = min(Vrange);			% voltage range of mouse clicks

	ipv = find(Vpfit >= Vmin);	
	imin = ipv(1);					% index of first data point within allowed voltage range
	ipv = find(Vpfit <= Vmax);
	imax = length(ipv);			% index of last data point within allowed voltage range

	VpTfit = Vpfit(imin:imax);
	yJTfit = yjprobe(imin:imax);	% data points used for Te-fit

	% fit straight line through data points in selected region
	[Pf,Sf] = polyfit(VpTfit,yJTfit,1);
	yJPf = polyval(Pf,VpTfit);
	plot(VpTfit,yJPf,'r')

	% now get the ion saturation current density
	title('Mouse select ion saturation current region','Color',[1,0,0],'FontSize',14)

	[Vrange,yJrange] = ginput(2);			% acccept graphical input from mouse

	Vmax = max(Vrange);
	Vmin = min(Vrange);			% voltage range of mouse clicks

    ipv = find(Vpfit >= Vmin);
	imin = ipv(1);					% index of first data point within allowed voltage range
	ipv = find(Vpfit <= Vmax);
	imax = length(ipv);			% index of last data point within allowed voltage range

	VpvJfit = Vpfit(imin:imax);
	yJJfit = yjprobe(imin:imax);	% data points used for Jsat-fit

	yJave = sum(yJJfit)/length(yJJfit);		% average data over allowed voltage range
	JsatMan(iramp) = exp(yJave) - Jmin0;	% Jsat estimated from data

	yJrange = [yJave yJave];		
	plot(Vrange,yJrange,'r')

	% Calculate plasma parameters resulting from fit
	TeMan(iramp) = 1/Pf(1);		% Pf(1) is the slope of the fit, d(logJ)/dV ~= 1/Te. Te is in [eV]
	Cs = 9.79e5*sqrt((Zion*TeMan(iramp)+Ti)/mi);	% ion sound speed [cm/s] 
	neMan(iramp) = abs(JsatMan(iramp))/(Amag*Cs*ee);	% electron density [cm^-3]
	Vf = (log(Jmin0)-Pf(2))/Pf(1);	% estimate plasma floating potential [V] from zero-crossing
	VsMan(iramp) = Vf + 2.3*TeMan(iramp);	% plasma space-charge potential [V]
    VsManC(iramp) = Vf + 2.3*TeMan(iramp)*Tescale; % corrected manual plasma space-charge potential [V]
    
   disp(' ')
   disp('Results of manual fit:')
   sst = sprintf('Te = %0.5g eV ; ne = %0.5g cm^-3 ; Vs = %0.5g V',TeMan(iramp),neMan(iramp),VsMan(iramp));
   disp(sst)
   disp(' ')
   A = input('Enter 0 to continue > ');
   if (A==0) % good fit, we're happy
       iramp = iramp + 1;
       if (iramp > iRampsMan(2))
           imandone = 1;
       end
   else % bad fit, we're not happy
       
   end
   hold off
   
end % while (imandone==0) % cycle through manual fits


% **************** Numerical fit to characteristic *******************

% choose which voltage ramps are to be fit numerically
disp(' ')
sst = ['Data consists of ',num2str(Nramps),' voltage sweeps'];
disp(sst)
disp(' ')
disp('Input indices of sweeps to be fit numerically in form [iStart,iEnd]:')
iRampsFit = input('(range [iStart,iEnd] must include points fit manually) ');
if (iRampsFit(1)<2); iRampsFit(1) = 2; end
if (iRampsFit(2)>Nramps-1); iRampsFit(2) = Nramps-1; end
% iRampsFit(1) = 1; iRampsFit(2) = 50;

myOptions = optimset('TolX',1e-5,'Tolfun',1e-5,'MaxIter',5000);  % convergence criteria for fminsearch 

% we'll assume that the point iRampsMan(1) is in the range iRampsFit(1):iRampsFit(2)
% First, scan down in iramp from iRampsMan(1) to iRampsFit(1)
for iramp = iRampsMan(1):-1:iRampsFit(1)
   
   if (JsatMan(iramp) ~= 0)	% an initial guess exists for this point;
      % use manual fit to data as initial guess for numerical fit
      TeFit(iramp) = TeMan(iramp); JsatFit(iramp) = JsatMan(iramp); VsFit(iramp) = VsMan(iramp); 
      VesatFit(iramp) = VsMan(iramp);
   else						% otherwise, use solution from previous point as initial guess
      TeFit(iramp) = TeFit(iramp+1); JsatFit(iramp) = JsatFit(iramp+1); VsFit(iramp) = VsFit(iramp+1);
      eslopeFit(iramp) = eslopeFit(iramp+1); VesatFit(iramp) = VesatFit(iramp+1);  JslopeFit(iramp) = JslopeFit(iramp+1);
   end
   if ((TeFit(iramp) < 0.01) | (TeFit(iramp) > 30))
       TeFit(iramp) = 1;
   end
	
   % Now, fit presumed functional form for probe characteristic (from
   % function 'Jpfit') to data 
   solV = [JsatFit(iramp) TeFit(iramp) VsFit(iramp) eslopeFit(iramp) ...
       VesatFit(iramp) JslopeFit(iramp)];	
   iStart = iStartRamp(iramp);
   iEnd = iEndRamp(iramp);
   if (iRampType > 1); iEnd = iStart + lambda - 1; end
   Vpfit = Vpv(iStart+1:iEnd-1);	% eliminate end points from fit because of occasional Jp spikes here
   Jfit = Jp(iStart+1:iEnd-1);
	[solV,fval,exitflag,solOutput] = fminsearch('Jpfit',solV,myOptions,Jfac,Vpfit,Jfit,iSheathExp,iFitEsat);
	VsFit(iramp) = solV(3); TeFit(iramp) = solV(2); JsatFit(iramp) = solV(1); eslopeFit(iramp) = solV(4);
   VesatFit(iramp) = solV(5); JslopeFit(iramp) = solV(6);
   
   sst = ['Numerically fitting iramp = ',num2str(iramp)];
   disp(' ');  disp(sst)
   sst = sprintf('Te = %0.5g eV ; Jsat = %0.5g A/cm^2 ; Vs = %0.5g V; Vesat = %0.5g V',...
      TeFit(iramp),JsatFit(iramp),VsFit(iramp),VesatFit(iramp));
   disp(sst)
   
end	% iramp = iRampsMan(1):-1:iRampsFit(1)   

% Now, scan up in iramp from iRampsMan(1)+1 to iRampsFit(2)
for iramp = (iRampsMan(1)+1):iRampsFit(2)
   
   if (neMan(iramp) ~= 0)  % if available, use manual fit to data as initial guess for numerical fit
      TeFit(iramp) = TeMan(iramp); JsatFit(iramp) = JsatMan(iramp); VsFit(iramp) = VsMan(iramp);
      VesatFit(iramp) = VsMan(iramp);
   else						% otherwise, use solution from previous point as initial guess
      TeFit(iramp) = TeFit(iramp-1); JsatFit(iramp) = JsatFit(iramp-1); VsFit(iramp) = VsFit(iramp-1);
      eslopeFit(iramp) = eslopeFit(iramp-1); VesatFit(iramp) = VesatFit(iramp-1); JslopeFit(iramp) = JslopeFit(iramp-1);
   end
   if ((TeFit(iramp) < 0.01) | (TeFit(iramp) > 30))
       TeFit(iramp) = 1;
   end
   
	% Now, fit presumed functional form for probe characteristic (from
	% function 'Jpfit') to data 
    solV = [JsatFit(iramp) TeFit(iramp) VsFit(iramp) eslopeFit(iramp) ...
        VesatFit(iramp) JslopeFit(iramp)];	  			
   iStart = iStartRamp(iramp);
   iEnd = iEndRamp(iramp);
   if (iRampType > 1); iEnd = iStart + lambda - 1; end
   Vpfit = Vpv(iStart+1:iEnd-1);	% eliminate end points from fit because of occasional Jp spikes here
   Jpfit = Jp(iStart+1:iEnd-1);
	[solV,fval,exitflag,solOutput] = fminsearch('Jpfit',solV,myOptions,Jfac,Vpfit,Jpfit,iSheathExp,iFitEsat);
	VsFit(iramp) = solV(3); TeFit(iramp) = solV(2); JsatFit(iramp) = solV(1); eslopeFit(iramp) = solV(4);
   VesatFit(iramp) = solV(5); JslopeFit(iramp) = solV(6);

   sst = ['Numerically fitting iramp = ',num2str(iramp)];
   disp(' ');  disp(sst)
   sst = sprintf('Te = %0.5g eV ; Jsat = %0.5g A/cm^2 ; Vs = %0.5g V ; Vesat = %0.5g V',...
		TeFit(iramp),JsatFit(iramp),VsFit(iramp),VesatFit(iramp));
   disp(sst)
   
end	% iramp = iRampsMan(1)+1:iRampsFit(2)     


% plot out data and fit to check on fit
figure
plot(Jp,'x')
ylabel('J-probe')
title('Probe current and fit')
hold on
TeFit = max(TeFit,0.1); % don't allow zero temperature
for iramp = iRampsFit(1):iRampsFit(2)
   iStart = iStartRamp(iramp);
   iEnd = iEndRamp(iramp);
   count = 0;
   JsatR(iramp) = 0;            % average over region with sufficiently negative tip voltage
   for ii = iStart:iEnd
       if (Vpv(ii) <= VJsatRmax)
           count = count + 1;
           JsatR(iramp) = JsatR(iramp) + Jp(ii);
       end
   end
   JsatR(iramp) = JsatR(iramp)/count;
   if (iRampType > 1); iEnd = iStart + lambda - 1; end
   Vpfit = Vpv(iStart:iEnd);
   Jpfit = JsatFit(iramp)*(1 - Jfac*exp((Vpfit-VsFit(iramp))/TeFit(iramp)));
   if (iSheathExp == 1)
        Jpfit = (JsatFit(iramp)+JslopeFit(iramp)*Vpfit).*(1 - Jfac*exp((Vpfit-VsFit(iramp))/TeFit(iramp)));
   end
   if (iFitEsat == 1) % fitting esat
      Nfit = iEnd - iStart + 1;
      Jesat = JsatFit(iramp)*(1 - Jfac*exp((VesatFit(iramp)-VsFit(iramp))/TeFit(iramp)));
      for ii = 1:Nfit
   	    if (Vpfit(ii) >= VesatFit(iramp))	% electron saturation current region
            Jpfit(ii) = Jesat + eslopeFit(iramp)*(Vpfit(ii)-VesatFit(iramp));
        end
      end
   end
   plot([iStart:iEnd],Jpfit,'r-');
end
axis([1 length(Jp) min(Jp) max(Jp)])

% calculate density and average Jsat
for iramp = iRampsFit(1):iRampsFit(2)
   Csfit = 9.79e5*sqrt((Zion*TeFit(iramp)+Ti)/mi);	% ion sound speed [cm/s]
   Jsat(iramp) = JsatFit(iramp) + JslopeFit(iramp)*(VsFit(iramp)-3*TeFit(iramp));  % use value well away from curve
   neFit(iramp) = -Jsat(iramp)/(Amag*ee*Csfit); % density [cm^-3]
   if (neFit(iramp) < 0) 
       neFit(iramp) = 0;
   end
   if (iFitEsat == 1)
      esatoisat(iramp) = Jfac*exp((VesatFit(iramp)-VsFit(iramp))/TeFit(iramp)) - 1;
   end
end

% make vector xFit which has average position at each iramp
xFit(1:Nramps) = 0;
for iramp = 1:Nramps
   iMean = round((iStartRamp(iramp) + iEndRamp(iramp))/2);	
   xFit(iramp) = xV(iMean);
end

% plot out plasma parameters vs. position
iStart = iRampsFit(1); iEnd = iRampsFit(2);
iStartM = iRampsMan(1); iEndM = iRampsMan(2);
xFitS = xFit(iStart:iEnd);
neFitS = neFit(iStart:iEnd);
TeFitS = TeFit(iStart:iEnd);
JsatS = Jsat(iStart:iEnd);
VsFitS = VsFit(iStart:iEnd);
JsatRS = JsatR(iStart:iEnd);
xManS = xFit(iStartM:iEndM);
JManS = JsatMan(iStartM:iEndM);
neManS = neMan(iStartM:iEndM);
TeManS = TeMan(iStartM:iEndM);
VsManS = VsMan(iStartM:iEndM);
VsManCS = VsManC(iStartM:iEndM);

figure
subplot(2,2,1)
plot(xFitS,neFitS/1e11,'bo')    % numerical fit
hold on
plot(xManS,neManS/1e11,'rx')    % manual fit
xlabel('position [cm]')
ylabel('n [10^{11} cm^{-3}]')
title('Density')
subplot(2,2,2)
plot(xFitS,TeFitS,'bo')     % numerical fit
hold on
plot(xManS,TeManS,'rx')     % manual fit
xlabel('position [cm]')
ylabel('T_e [eV]')
title('Temperature')
subplot(2,2,3)
plot(xFitS,VsFitS,'bo')     % numerical fit
hold on
plot(xManS,VsManS,'rx')     % manual fit
xlabel('position [cm]')
ylabel('V_s [V]')
title('Space-charge potential')
subplot(2,2,4)
plot(xFitS,JsatS,'bo')      % numerical fit to Jsat
hold on
plot(xManS,JManS,'rx')      % manual fit
plot(xFitS,JsatRS,'gd')     % robust Jsat fit
xlabel('position [cm]')
ylabel('Jsat [A/cm^2]')
title('ion saturation current density')

% mystring = sprintf('save ProbeFits%0.4i xFitS neFitS TeFitS JsatS VsFitS AreaP xScale Rc JsatRS xManS neManS TeManS VsManS VsManCS JManS',shot);
% eval(mystring);

