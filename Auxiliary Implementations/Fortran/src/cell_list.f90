! celllist(:, :, :) 1st dim tells how many (max 9) and which bacteria are in
! the cell, 2nd and 3rd dim are ii and jj indices of the cell.
! ipc(:, :) ith bacteria cell - tells in which cell the given bacteria is and on which position in a celllist chain in that cell.


subroutine make_cell_list(nbact, maxnbact, rb, celllist, ipc, ncells, soc, mycoords)
  implicit none

  integer, intent(in) :: ncells(2), nbact, maxnbact, mycoords(2)
  real, intent(in) :: soc, rb(2, maxnbact)

  integer, intent(out) :: celllist(200, ncells(1), ncells(2)), ipc(3, maxnbact)

  integer :: ii, jj

  ! rb & vb have the same origin for all tasks, celllist & ipc has unique
  ! origin in the top left corner for each task

  celllist(:, :, :) = 0
  ipc(:, :) = 0

  do ii = 1, nbact
     ipc(1:2, ii) = 1 + floor(rb(:, ii) / soc) - mycoords(:) * (ncells(:) - 2) + 1

     celllist(1, ipc(1, ii), ipc(2, ii)) = celllist(1, ipc(1, ii), ipc(2, ii)) + 1
     celllist((celllist(1, ipc(1, ii), ipc(2, ii)) + 1), ipc(1, ii), ipc(2, ii)) = ii
     ipc(3, ii)=celllist(1, ipc(1, ii), ipc(2, ii))
  enddo
end subroutine make_cell_list
