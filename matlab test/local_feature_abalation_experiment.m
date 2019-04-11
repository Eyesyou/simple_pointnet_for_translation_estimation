clear all
a=csvread('run_with5-tag-pos_dis.csv',1,1);
b=csvread('run_without5-tag-pos_dis.csv',1,1);
c=csvread('run_with5-tag-ang_dis.csv',1,1);
d=csvread('run_without5-tag-ang_dis.csv',1,1);
a_mean = mean(a(:,2))*ones(100,1);
b_mean = mean(b(:,2))*ones(100,1);


fig = figure;
left_color = [0 0.5 1];
right_color = [0.8 0.1 0.2];
set(fig,'defaultAxesColorOrder',[left_color; right_color]);

set(gca,'DefaultTextFontSize',50, 'yColor',left_color)
set(gcf,'Position',[100 100 2400 1200])
yyaxis left
l1=plot(a(:,1),a(:,2),'MarkerEdgeColor',left_color,'linewidth',3);
hold on
l2=plot(b(:,1),b(:,2),':','MarkerEdgeColor',left_color,'linewidth',3);
ylabel('translation loss', 'FontSize', 25,'Color',left_color);
ax = gca;
ax.FontSize = 25; 
legend([l1 l2] , {'with local features','without local features'},'Location','northwest')

plot(a(:,1),a_mean,'-','MarkerEdgeColor',left_color,'linewidth', 1)
plot(b(:,1),b_mean,':','MarkerEdgeColor',left_color,'linewidth', 2)
text(23,a_mean(1)+0.01,'mean','fontsize',25)
text(65.5,b_mean(1)+0.01,'mean','fontsize',25)

yyaxis right
l3=plot(c(:,1),c(:,2),'MarkerEdgeColor',right_color,'linewidth',3);
hold on
l4=plot(d(:,1),d(:,2),':','MarkerEdgeColor',right_color,'linewidth',3);
r_x=gca;

legend([l1 l2 l3 l4] , {'translation loss without local features','translation loss with local features','quaternion loss without local features','quaternion loss with local features'},'TextColor','black','Location','northeast')
ylabel('quaternion loss', 'FontSize', 25);
axis([0,100, 0,0.1])
%title('Transformation loss comparison between networks with and without local features', 'FontSize', 25);

