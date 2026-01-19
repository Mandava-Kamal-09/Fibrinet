%% ABM code to simulate simple stochastic dynamics of fibrin monomers
%%  moving through a domain and polymerizing if certain conditions are met

%% Last updated: 4/11/23

close all
clear all


run_number = 1; %keep at 1 for indiviaul runs


% Set discretization
%
dt     = 0.005;                  % temporal discretization
Tend   = 10.0;                   % end time (a.u.)
Nt     = Tend/dt;                % number of timesteps
dx     = 0.045;                  % length of fibrin monomer (um)
tplot  = 10;                     % visualize every 10 time units


% Set fibrin filament parameters
%
ppoly0   = 0.324*dt/dx;  % probability to polymerize. FIND VALUE FOR FIBRIN!!!!!
bindangle = 20*pi/180; % angle range (plus/minus) at which monomer must meet other monomer or polymer in order to bind
Nmonomers0 = 100;       % number of fibrin monomers in the system
Npolymers0 = 0; %number of fibrin polymers in system
rotate_coeff = 0.01; %rotation coefficient
diff_coeff = 0.1; %diffusion coefficient
thres_dist = 0.02; %threshold distance, in um, monomers must be in order to bind to others


tic
for stats=1:run_number
    rng(stats,'twister')

    Nmonomers=Nmonomers0;
    Npolymers=Npolymers0;
    
    % Initiailze
    pos_mon = rand(Nmonomers,2); %initialize monomer positions randomly
    theta_mon = 2*pi*rand(Nmonomers,1); %initialize monomer orientations randomly
    
    pos_poly = rand(Npolymers,2); %initialize polymer positions randomly
    theta_poly = 2*pi*rand(Npolymers,1); %initialize polymer orientations randomly
    len_poly = zeros(Npolymers,1); %initialize polymer lengths at 0
    
    %IDEA: all monomers have length 0.045 um, where their position given in
    %pos_mon is the (x,y) coordinate of one of their ends. The other end is
    %0.045 um away with angle theta_mon. For each polymer we create, we
    %need to associate with it a length (it's the sum of the 2 entities
    %that created it minus 22.5 to account for the overlap. e.g., if 2
    %monomers come together, new polymer is length
    %0.045+0.045-0.0225=0.0675. If a polymer of length 0.0675 and a monomer
    %come together, new polymer is legnth 0.045+0.0675-0.0225=0.9
    
    if stats==1 %only make a movie for the first run
        % following 4 pieces are for saving network movie
        w = VideoWriter('monomer_movie.avi');
        w.FrameRate = 10;
        w.Quality = 100;
        open(w);
        % end variables for network movie
    end
    
     
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %                    Run microscale simulation
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %tic
    
    %BB: TO ADD IN STOCHASTIC RATES, I'LL NEED THE IF ANGLE < BINDANGLE
    %LOOPS TO INCLUDE RANDOM NUMBERS AND THEN ONLY ALLOW POLYMERIZATION IF
    %RANDOM NUMBER IS LESS THAN BINDING PROBABILITY
    for tau=1:Nt
        
        % For each monomer, allow it to diffuse and/or rotate its
        % orientation
        
        % Update positions and orientations of all monomers and polymers
        pos_mon = pos_mon + diff_coeff*sqrt(dt)*randn(Nmonomers,2);
        theta_mon = theta_mon + rotate_coeff*sqrt(dt)*randn(Nmonomers, 1);
        
        pos_poly = pos_poly + diff_coeff*sqrt(dt)*randn(Npolymers,2);
        theta_poly = theta_poly + rotate_coeff*sqrt(dt)*randn(Npolymers, 1);
        
        pos_mon_temp=pos_mon;
        pos_poly_temp=pos_poly;
        theta_mon_temp=theta_mon;
        theta_poly_temp=theta_poly;
        Npolymers_temp=Npolymers;
        Nmonomers_temp=Nmonomers;
        len_poly_temp=len_poly;
        
        count=0;
        countind=0;
        
        %create matrix of distances between monomers, and angles between
        %monomrs
        for i=1:Nmonomers
            for j=1:Nmonomers
                Dist_pos_mon(i,j)=norm(pos_mon(i,:)-pos_mon(j,:));
                angle_mon(i,j)=abs(theta_mon(i)-theta_mon(j));
                if angle_mon(i,j)>pi
                   angle_mon(i,j)=2*pi-angle_mon(i,j); 
                end                
            end
        end
        
        %find which monomers are within threshold distance and angle of each other
        for i=1:Nmonomers
            ind=find(Dist_pos_mon(:,i)<thres_dist & angle_mon(:,i)<bindangle);
            if length(ind)>1
                possible_monomers(i,1:length(ind))=ind;
            else
                possible_monomers(i,1:2)=0;
            end
            ind=0;
        end
        
        %find the indices of the nonzero entries of the possible_monomers matrix  
        indx=find(possible_monomers(:,1)>0); 
        
        %make new matrix of just nonzero entries. first column gives
        %monomer number; subsequent columns give other monomers that are
        %close to column 1's monomer
        for i=1:length(indx)
            choices(i,:)=possible_monomers(indx(i),:);
        end
        
        choices_sort=choices;
        
        %randomly choose which monomers polymerize. Once a decision has
        %been made, reset the other choices entries containing those
        %monomer numbers to 0
        for i=1:length(indx)
            %sort "choices" so that each row is always nonzero entries
            %followed by 0 entries
            choices_sort=sort(choices_sort,2);
            if choices_sort(i,1)>0 %only work on monomers that haven't already polymerized
                indy=find(choices_sort(i,:)>0); %only consider "choices" entries greater than 0
                ind_1(i)=randi([2 length(indy)]); %pick a random entry, not including the 1st entry, since that is just the currenet monomer, which can't polymerize with itself
                mon_choice(i)=choices_sort(i,ind_1(i)); %find the monomer number that will polymerize with monomer choices(i,1)
                
                mon1=choices_sort(i,1);
                mon2=mon_choice(i);
                
                %polymerize these 2 monomers
                Npolymers_temp=Npolymers_temp+1; %number of polymers increases by one
                Nmonomers_temp=Nmonomers_temp-2; %number of monomers decreases by two
                
                %remove the 2 monomers
                %set the mon1=choices(i,1) and mon2=mon_choice(i) entries of these vectors to 600,
                %and after all the loops, we'll remove all rows
                %that have entries of 600
                pos_mon_temp(mon1,:)=600;
                pos_mon_temp(mon2,:)=600;
                theta_mon_temp(mon1)=600;
                theta_mon_temp(mon2)=600;
                
                count=count+1;
                
                %calculate positions, angle, and length of new
                %polymer
                pos_poly_temp(Npolymers+count,:)=(pos_mon(mon1,:)+pos_mon(mon2,:))/2; %add a row to the pos_poly_temp matrix, because we have a new polymer created
                theta_poly_temp(Npolymers+count,1)=(theta_mon(mon1)+theta_mon(mon2))/2; %set new polymer angle to be average of angles
                len_poly_temp(Npolymers+count)=dx+dx/2; %length is 2 monomers minus half a monomer
                
                %set to 0 all other "choices" entries equal to these 2
                %monomers, since they can no longer polymerize with other
                %monomers
                for j=1:length(indx)
                    for k=1:length(indy)
                        if choices_sort(j,k)==mon1 || choices_sort(j,k)==mon2
                            choices_sort(j,k)=0;
                        end
                    end
                end
            end
        end %end for i
        
        pos_mon=pos_mon_temp;
        theta_mon=theta_mon_temp;
        
        %now delete all rows of pos_mon_temp and theta_mon_temp arrays that have 600 entries
        ind600=find(pos_mon_temp(:,1)==600);
        pos_mon([ind600],:)=[];
        theta_mon([ind600])=[];
        
        %         for i=1:Nmonomers
        %             if pos_mon_temp(i,1)==600
        %                 pos_mon(i,:)=[];
        %                 theta_mon(i)=[];
        %             end
        %         end
        
        
        Nmonomers=Nmonomers_temp;
        Npolymers=Npolymers_temp;
        
        pos_poly=pos_poly_temp;
        theta_poly=theta_poly_temp;
        len_poly=len_poly_temp;
        
        choices=[];
        indx=[];
        indy=[];
        ind_1=[];
        mon_choice=[];
        mon2=0;
        mon1=0;
        
        
        %START JUST WITH MONOMERS POLYMERIZING WITH EACH OTHER, AND DIMERS
        %MOVING. PLOT THESES RESULTS. THEN BUILD IN MONOMER-POLYMER AND
        %POLYMER-POLYMER INTERACTIONS
        
%I NEED TO FIGURE OUT HOW TO RESTRICT MONOMERS AND POLYMERS TO A SPECIFICED
%REGION (I.E., HAVE BOUNDARIES), AND THEN PLOT JUST THAT REGION.

%I ALSO NEED TO PLOT THE MONOMERS AS LITTLE LINE SEGMENTS WITH SPECIFIED
%LENGTH AND ORIENTATION, AND I NEED TO PLOT THE POLYMERS AS BIGGER LINE
%SEGMENTS WITH SPECIFIED LENGTH AND ORIENTATION. PLOT POLYMERS IN RED,
%MONOMERS IN BLACK, PERHAPS
        
        figure(2);
        if mod(tau,tplot)==0
        for j=1:Nmonomers
            plot([pos_mon(j,1) pos_mon(j,1)+dx*cosd(theta_mon(j)*180/pi)],[pos_mon(j,2) pos_mon(j,2)+dx*sind(theta_mon(j)*180/pi)],'k','LineWidth',2)
            %plot(pos_mon(j,1),pos_mon(j,2),'ko','MarkerFaceColor','k')
            hold on
        end
        for j=1:Npolymers
            plot([pos_poly(j,1) pos_poly(j,1)+len_poly(j)*cosd(theta_poly(j)*180/pi)],[pos_poly(j,2) pos_poly(j,2)+len_poly(j)*sind(theta_poly(j)*180/pi)],'r','LineWidth',2)
            hold on
        end
        xlim([-0.3 1.1]); ylim([-0.3 1.1]);
        title(['time = ',num2str(tau*dt)]);
        set(gca,'fontname','times','fontsize',30); box on;
        drawnow;
        hold off;
        frame = getframe(gcf);
        writeVideo(w,frame);
        end
        
        %BB: PROBLEM. DOING IT THIS WAY PRIORITIZES MONOMER-MONOMER
        %POLYMERIZATION OVER MONOMER-POLYMER POLYMERIZATION. IF MONOMER 3,
        %E.G. IS NEAR MONOMER 67 AND POLYMER 4, SINCE I DO MONOMER-MONOMER
        %INTERACTIONS FIRST, IT WILL POLYMERIZE WITH MONOMER 67 RATHER THAN
        %POLYMER 4. OHHHH, BUT IF I GIVE THIS A PROBABILITY, RATHER THAN
        %JUST 100% OF THE TIME IT MEETS THE THRESHOLD DOING IT, THEN MAYBE
        %THAT'S OKAY? EITHER WAY, PROBABLY "GOOD ENOUGH" FOR PRELIMINARY
        %DATA
        
        
    end %end tau
end %end stats
toc
close(w) %for movie


        
% %         %find how many monomers each monomer is close to
% %         for i=1:Nmonomers
% %             num_close(i)=length(find(indmat==i));
% %         end
% %            
% %                 %if monomers are within threshold distance of each other,
% %                 %check if their angles are such that polymerization would
% %                 %be possible
% %                 if Dist_pos_mon(i,j)< thres_dist && Dist_pos_mon(i,j)>0
% %                     angle = abs(theta_mon(i)-theta_mon(j));
% %                     if angle>pi
% %                         angle=2*pi-angle; %BB: DO WE NEED THIS???
% %                     end
% %                     if angle < bindangle %check for angle threshold
% %                         countind=countind+1;
% %                         num_close(countind,1)=i;
% %                         num_close(countind,2)=j;
% %                     end %if angle < bindangle
% %                 end
% %             end
% %         end
%         
% %         %create matrix of distances between monomers
% %         for i=1:Nmonomers
% %             for j=1+i:Nmonomers
% %                 Dist_pos_mon(i,j)=norm(pos_mon(i,:)-pos_mon(j,:));
% %                 %if monomers are within threshold distance of each other,
% %                 %check if their angles are such that polymerization would
% %                 %be possible
% %                 if Dist_pos_mon(i,j)< thres_dist && Dist_pos_mon(i,j)>0
% %                     angle = abs(theta_mon(i)-theta_mon(j));
% %                     if angle>pi
% %                         angle=2*pi-angle; %BB: DO WE NEED THIS???
% %                     end
% %                     if angle < bindangle %check for angle threshold
% %                         countind=countind+1;
% %                         num_close(countind,1)=i;
% %                         num_close(countind,2)=j;
% %                     end %if angle < bindangle
% %                 end
% %             end
% %         end
%         
%         %num_close array is the list of monomers that are close enough, and
%         %have the correct angle, to other monomers that polymerization
%         %could happen this timestep. The number of rows is how many pairs
%         %of monomers are close enough. Each column contains 
%         
%         %find the indices of the matrix corresponding to entries that are
%         %less than thres_dist
%         [indx,indy]=find(Dist_pos_mon<thres_dist & Dist_pos_mon>0); 
%         indmat=[indx,indy];
%         
%         %find how many monomers each monomer is close to
%         for i=1:Nmonomers
%             num_close(i)=length(find(indmat==i));
%         end
%         
%         %for each monomer that is close to 1 or more, randomly choose which
%         %monomer it polymerizes with
%         %THIS WILL STILL BE BIASED BECAUSE MONOMERS NEAR ONE OTHER WILL
%         %POLYMERIZE BEFORE MONOMERS NEAR MORE THAN 1
%         for i=1:Nmonomers
%             if num_close(i)==1
%                 
%             elseif num_close(i)>1
%                 
%             end %end if num_close(iO>0
%         end %end for i=1:Nmonomers
%         
%                 
%         %BB: I need a way to account for, say, monomer 7 is close to
%         %monomers 15 and 88, but monomer 88 is close to monomer 90. So 7
%         %has to choose between 15 and 88, but 88 has to choose between 7
%         %and 90. How to make that decision?? 
%         
%             
%         
% %         %sort them by row number so we can see which monomers are close to
% %         %multiple monomers
% %         indmat_sort=sortrows(indmat);
%         
%         
%         %BB: PROBLEM WITH MONOMER LOOP; ONE MONOMER COULD BE WITHIN
%         %PROXIMITY OF MULTIPLE MONOMERS. AS IS, IT FORMS POLYMER WITH ALL
%         %OF THEM, WHICH ISN'T PHYSICAL. THIS WILL BE ISSUE ON POLYMER
%         %LOOPS, TOO
%         % Check for binding of monomers to other monomers
%         for j=1:Nmonomers %loop over number of monomers
%             for k=1+j:Nmonomers %compare to other monomers
%                 if (norm(pos_mon(j,:) - pos_mon(k,:)) < thres_dist) %check for proximity
%                     angle = abs(theta_mon(j)-theta_mon(k));
%                     if angle>pi
%                         angle=2*pi-angle; %BB: DO WE NEED THIS???
%                     end
%                     if angle < bindangle %check for angle threshold
%                         %update number of polymers and monomers
%                         Npolymers_temp=Npolymers_temp+1; %number of polymers increases by one
%                         Nmonomers_temp=Nmonomers_temp-2; %number of monomers decreases by two
%                         
%                         %remove the 2 monomers
%                         %set the j and k entries of these vectors to 600,
%                         %and after all the loops, we'll remove all rows
%                         %that have entries of 600
%                         pos_mon_temp(j,:)=600;
%                         pos_mon_temp(k,:)=600;
%                         theta_mon_temp(j)=600;
%                         theta_mon_temp(k)=600;
%                         
%                         count=count+1;
%                       
%                         %calculate positions, angle, and length of new
%                         %polymer
%                         pos_mon_temp(Npolymers+count,:)=(pos_mon(j,:)+pos_mon(k,:))/2; %add a row to the pos_mon_temp matrix, because we have a new polymer created
%                         theta_poly_temp(Npolymers+count)=(theta_mon(j)+theta_mon(k))/2; %set new polymer angle to be average of angles
%                         len_poly_temp(Npolymers+count)=dx+dx/2; %length is 2 monomers minus half a monomer
%                         
%                         %COULD I CALCULATE MATRIX OF DISTANCES BETWEEN
%                         %EVERY MONOMER AND EVERY OTHER MONOMER, and then
%                         %just work on the ones that have small enough
%                         %distances (rather than doing all these loops)?
%                         
%                     end
%                 end
%             end
%         end
%         % Check for binding of polymers to monomers or other polymers
%         for i=1:Npolymers %loop over number of polymers
%             for k=1:Nmonomers %compare to all monomers
%                 if (norm(pos_poly(i,:) - pos_mon(k,:)) < thres_dist) %check for proximity
%                     angle = abs(theta_poly(i)-theta_mon(k));
%                     if angle>pi
%                         angle=2*pi-angle;
%                     end
%                     if angle < bindangle %check for angle threshold
%                         %update number of monomers (number of polymers is
%                         %unchanged)
%                         Nmonomers_temp=Nmonomers_temp-1; %number of monomers decreases by 1
%                         
%                         %update positions of monomer and polymer, and
%                         %length of polymer
%                         pos_mon_temp(k,:)=600; %remove monomer
%                         theta_mon_temp(k)=600;
%                         pos_poly_temp(i,:)=(pos_poly(i,:)+pos_mon(k,:))/2; %set new polymer position to be average of positions
%                         theta_poly_temp(i)=(theta_poly(i)+theta_mon(k))/2; %set new polymer angle to be average of angles
%                         len_poly_temp(i)=len_poly_temp(i)+dx/2; %new length is old length plus half a monomer
%                     end
%                 end
%             end
%             for j=1+i:Npolymers %compare to all polymers
%                 if (norm(pos_poly(i,:) - pos_poly(j,:)) < thres_dist) %check for proximity
%                     angle = abs(theta_poly(i)-theta_poly(j));
%                     if angle>pi
%                         angle=2*pi-angle;
%                     end
%                     if angle < bindangle %check for angle threshold
%                         %update number of polymers
%                         Npolymers_temp=Npolymers_temp-1; %number of polymers decreases by 1
%                         
%                         %update positions and lengths of polymers
%                         pos_poly_temp(i,:)=(pos_poly(i,:)+pos_poly(j,:))/2; %set new polymer position to be average of positions
%                         theta_poly_temp(i)=(theta_poly(i)+theta_poly(j))/2; %set new polymer angle to be average of angles
%                         len_poly_temp(i)=len_poly(i)+len_poly(j)-dx/2; %new length is sum of 2 polymers minus half a monomer
%                         pos_poly_temp(j,:)=600; %remove the second polymer
%                         theta_poly_temp(j)=600; %remove second polymer
%                         len_poly_temp(j)=600; %remove second polymer
%                     end
%                 end
%             end %j loop
%         end %i loop
%         
%         % Update position and theta arrays
%         
%         %find the positions of the 600 entries in the various arrays
%         ind_pos_mon=find(pos_mon_temp(:,1)==600);
%         ind_pos_poly=find(pos_poly_temp(:,1)==600);
%         ind_theta_mon=find(theta_mon_temp(:)==600);
%         ind_theta_poly=find(theta_poly_temp(:)==600);
%         ind_len_poly=find(len_poly_temp(:)==600);
%         
%         %remove all rows that have 600 entries
%         pos_mon_temp(ind_pos_mon,:)=[];
%         pos_poly_temp(ind_pos_poly,:)=[];
%         theta_mon_temp(ind_theta_mon)=[];
%         theta_poly_temp(ind_theta_poly)=[];
%         len_poly_temp(ind_len_poly)=[];
%         
%         %set arrays to be new values
%         pos_mon=pos_mon_temp;
%         pos_poly=pos_poly_temp;
%         theta_mon=theta_mon_temp;
%         theta_poly=theta_poly_temp;
%         len_poly=len_poly_temp;
%         
%         Nmonomers=Nmonomers_temp;
%         Npolymers=Npolymers_temp;
%           
% 
%     end %end tau
% end %end stats
% % close(w) %for network movie
% toc
 
% %change the name of the file below for each inidiviaul run:
% save -ascii Fibrin_array_test.dat Fibrin_array_T